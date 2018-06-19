#########################
# Expression Predicates #
#########################

is_call(x) = Base.Meta.isexpr(x, :call) && (x.args[1] !== GlobalRef(Core, :tuple))

is_macro(x, name) = Base.Meta.isexpr(x, :macrocall) && x.args[1] == name

function is_method_definition(x)
    if isa(x, Expr)
        if x.head == :function
            return true
        elseif x.head == :(=) && isa(x.args[1], Expr)
            lhs = x.args[1]
            if lhs.head == :where
                lhs = lhs.args[1]
            end
            return lhs.head == :call
        end
    end
    return false
end

############################
# Expression Match/Replace #
############################

function replace_match!(replace, ismatch, x)
    if ismatch(x)
        return replace(x)
    elseif isa(x, Array)
        for i in eachindex(x)
            x[i] = replace_match!(replace, ismatch, x[i])
        end
    elseif isa(x, Expr)
        replace_match!(replace, ismatch, x.args)
    end
    return x
end

############
# Julia IR #
############

#=== reflection ===#

mutable struct Reflection
    signature::DataType
    method::Method
    static_params::Vector{Any}
    code_info::CodeInfo
end

# Return `Reflection` for signature `sigtypes` and `world`, if possible. Otherwise, return `nothing`.
function reflect(@nospecialize(sigtypes::Tuple), world::UInt = typemax(UInt))
    # This works around a subtyping bug. Basically, callers can deconstruct upstream
    # `UnionAll` types in such a way that results in a type with free type variables, in
    # which case subtyping can just break.
    #
    # God help you if you try to use a type parameter here (e.g. `::Type{S} where S<:Tuple`)
    # instead of this nutty workaround, because the compiler can just rewrite `S` into
    # whatever it thinks is "type equal" to the actual provided value. In other words, if
    # `S` is defined as e.g. `f(::Type{S}) where S`, and you call `f(T)`, you should NOT
    # assume that `S === T`. If you did, SHAME ON YOU. It doesn't matter that such an
    # assumption holds true for essentially all other kinds of values. I haven't counted in
    # a while, but I'm pretty sure I have ~40+ hellish years of Julia experience, and this
    # still catches me every time. Who even uses this crazy language?
    S = Tuple{map(s -> Core.Compiler.has_free_typevars(s) ? typeof(s.parameters[1]) : s, sigtypes)...}
    (S.parameters[1]::DataType).name.module === Core.Compiler && return nothing
    _methods = Base._methods_by_ftype(S, -1, world)
    length(_methods) == 1 || return nothing
    type_signature, raw_static_params, method = first(_methods)
    method_instance = Core.Compiler.code_for_method(method, type_signature, raw_static_params, world, false)
    method_signature = method.sig
    static_params = Any[raw_static_params...]
    code_info = Core.Compiler.retrieve_code_info(method_instance)
    isa(code_info, CodeInfo) || return nothing
    code_info = Core.Compiler.copy_code_info(code_info)
    return Reflection(S, method, static_params, code_info)
end

#=== `substitute!` replacement ===#

# This function is essentially a half-port of the old `Core.Compiler.substitute!` function
# that works on pre-inference IR and only does the things Cassette needs it to do
function substitute_static_params_and_offset_slots!(@nospecialize(e),
                                                    @nospecialize(spsig),
                                                    spvals::Vector{Any},
                                                    offset::Int)
    if isa(e, Core.SlotNumber)
        return Core.SlotNumber(e.id::Int + offset)
    end
    if isa(e, Core.NewvarNode)
        return Core.NewvarNode(substitute_static_params_and_offset_slots!(e.slot, spsig, spvals, offset))
    end
    if isa(e, Expr)
        head = e.head
        if head === :static_parameter
            return QuoteNode(spvals[e.args[1]])
        elseif head === :cfunction
            @assert !isa(spsig, UnionAll) || !isempty(spvals)
            if !(e.args[2] isa QuoteNode) # very common no-op
                e.args[2] = substitute_static_params_and_offset_slots!(e.args[2], spsig, spvals, offset)
            end
            e.args[3] = ccall(:jl_instantiate_type_in_env, Any, (Any, Any, Ptr{Any}), e.args[3], spsig, spvals)
            e.args[4] = svec(Any[ccall(:jl_instantiate_type_in_env, Any, (Any, Any, Ptr{Any}), argt, spsig, spvals)
                                 for argt in e.args[4]]...)
        elseif head === :foreigncall
            @assert !isa(spsig, UnionAll) || !isempty(spvals)
            for i = 1:length(e.args)
                if i == 2
                    e.args[2] = ccall(:jl_instantiate_type_in_env, Any, (Any, Any, Ptr{Any}), e.args[2], spsig, spvals)
                elseif i == 3
                    e.args[3] = svec(Any[ccall(:jl_instantiate_type_in_env, Any, (Any, Any, Ptr{Any}), argt, spsig, spvals)
                                         for argt in e.args[3]]...)
                elseif i == 4
                    @assert isa((e.args[4]::QuoteNode).value, Symbol)
                elseif i == 5
                    @assert isa(e.args[5], Int)
                else
                    e.args[i] = substitute_static_params_and_offset_slots!(e.args[i], spsig, spvals, offset)
                end
            end
        elseif !(Base.Meta.is_meta_expr_head(head))
            substitute_static_params_and_offset_slots!(e.args, spsig, spvals, offset)
        end
    end
    if isa(e, Array)
        for i = 1:length(e)
            e[i] = substitute_static_params_and_offset_slots!(e[i], spsig, spvals, offset)
        end
    end
    return e
end

#=== IR Repair ===#

# TODO: update this
function fix_labels_and_gotos!(code::Vector)
    changes = Dict{Int,Int}()
    for (i, stmnt) in enumerate(code)
        if isa(stmnt, LabelNode)
            code[i] = LabelNode(i)
            changes[stmnt.label] = i
        end
    end
    for (i, stmnt) in enumerate(code)
        if isa(stmnt, GotoNode)
            code[i] = GotoNode(get(changes, stmnt.label, stmnt.label))
        elseif Base.Meta.isexpr(stmnt, :enter)
            stmnt.args[1] = get(changes, stmnt.args[1], stmnt.args[1])
        elseif Base.Meta.isexpr(stmnt, :gotoifnot)
            stmnt.args[2] = get(changes, stmnt.args[2], stmnt.args[2])
        end
    end
    return code
end

#################
# Miscellaneous #
#################

unquote(x) = x
unquote(x::QuoteNode) = x.value
unquote(x::Expr) = x.head == :quote ? first(x.args) : x

function unqualify_name(e::Expr)
    @assert e.head == :(.)
    return unqualify_name(last(e.args))
end

unqualify_name(name::Symbol) = name

# define safe loggers for use in generated functions (where task switches are not allowed)
for level in [:debug, :info, :warn, :error]
    @eval begin
        macro $(Symbol("safe_$level"))(ex...)
            macrocall = :(@placeholder $(ex...))
            # NOTE: `@placeholder` in order to avoid hard-coding @__LINE__ etc
            macrocall.args[1] = Symbol($"@$level")
            quote
                old_logger = global_logger()
                global_logger(Logging.ConsoleLogger(Core.stderr, old_logger.min_level))
                ret = $(esc(macrocall))
                global_logger(old_logger)
                ret
            end
        end
    end
end
