mutable struct IDIndex
    persisted::Dict{UInt,Any}
    idcount::UInt
    IDIndex() = new(Dict{UInt,Any}(), 1)
end

function generate_id!(i::IDIndex, x)
    if isimmutable(typeof(x))
        i.idcount += 1
        return i.idcount
    else
        return UInt(pointer_from_objref(x))
    end
end

# TODO: need the array version of this
function persist_ids!(i::IDIndex, item::Tagged, fields::Tuple)
    # make a NamedTuple from `item`'s value type where each field points to
    # the corresponding ID from `fields` (use `0` as sentinel value)
end

# TODO: optimization pass over the IR that sniffs for `getfield`s/`getindex`s and prefetches
# the indexed shadow structure for the corresponding variable. What access-level depth could
# this reasonably handle?

# IDEA: What if every primal SSAValue had an associated ID slot, and the ID slot values
# could propagate between call boundaries/variable scopes? Could that be a way of getting
# rid of the index data structure?

struct Tagged{T<:AbstractTag,V}
    tag::T
    value::V
    id::UInt
end

istagged(::Any, ::AbstractContext) = false
istagged(::Tagged{T}, ::AbstractContext{<:AbstractPass,T}) where {T<:AbstractTag} = true

tag(x, ctx::AbstractContext) = Tagged(ctx.tag, x, generate_id!(ctx.ids, x))

untag(x, ctx::AbstractContext) = istagged(x, ctx) ? x.value : x

@generated function _new_tagged(ctx::C, ::Type{T}, fields...) where {C<:Context,T}
    if # none of the args are Tagged
        return quote
            $(Expr(:meta, :inline))
            $(Expr(:new, T, Expr(:..., :fields)))
        end
    else # we need to tag the output and persist the fields' ids in the IDIndex
        return quote
            result = $(Expr(:new, T, [:(untag(fields[$i], ctx)) for i in 1:nfields(args)]...))
            tagged_result = tag(result, ctx)
            persist_ids!(ctx.ids, tagged_result, fields)
            return tagged_result
        end
    end
end
