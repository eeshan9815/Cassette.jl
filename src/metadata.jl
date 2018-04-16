##############
# MetaMemory #
##############

struct MetaMemory
    memory::Dict{UInt,Any}
    MetaMemory() = new(Dict{UInt,Any}())
end

@generated function address(x::T) where {T}
    if T.mutable
        return quote
            $(Expr(:meta, :inline))
            UInt(pointer_from_objref(x))
        end
    else
        return :(error("MetaMemory cannot be addressed by immutable objects"))
    end
end

@inline function Base.getindex(m::MetaMemory, ::C, x::X) where {C<:Context,X}
    _, y = m.memory[address(x)]::Tuple{C,metatype(C, X)}
    return y
end

@inline function Base.setindex!(m::MetaMemory, y, ctx::C, x::X) where {C<:Context,X}
    @assert isa(y, metatype(C, X))
    m.memory[address(x)] = (ctx, y)
    finalizer(x -> delete!(m.memory, x), x)
    return y
end

#=
Cassette maintains its own global metamemory that is used by default for all contextual
execution. This is important because the memory that is being "shadowed" is, by necessity,
Julia's global memory.

For example, imagine a scenario where two separate threads executed code in two different
Cassette contexts, but accessed the same global const array. As it stands, sthis array has
a single address in our metamemory space, so changes to this array from either thread will
de facto propagate to the same corresponding array in metamemory. Cassette can then
automatically prevent one context from accidentally mutating metadata stored by the other
context via a simple type check (see implementation above).

Imagine, instead, the outcome if each context had maintained its own trace-local metamemory.
The result would be that the mutation of the original array from within one context could
easily invalidate the metadata w.r.t. that array stored by the other context. For example,
imagine that the two contexts in our scenario were created for the purpose of automatic
differentiation. If the array value was mutated from within one context, that context
could perform a corresponding update to a derivative array. However, the other context
wouldn't see the change, and thus (presumably) would then be storing an invalid
derivative array w.r.t. the original array's value.
=#
const MEMORY = MetaMemory()

########
# Meta #
########

struct Meta{D,T<:Union{NamedTuple,Array}}
    id::Int
    substructure::Union{T,Unused}
end

@inline metadatatype(::Type{<:Context}, ::DataType) = Unused

_ndims(::Type{Array{T,N}}) where {T,N} = N
_ndims(::Type{Array{T,N} where T}) where {N} = N
_ndims(::Type{Array}) = -1
_ndims(::Type{Array{T,N} where N}) where {T} = -1

@generated function metatype(::Type{C}, ::Type{T}) where {C<:Context,T}
    if T <: Array
        E, N = eltype(T), _ndims(T)
        if N == -1
            result = :(Meta{metadatatype(C, T),Array{metatype(C, $E)}})
        else
            result = :(Meta{metadatatype(C, T),Array{metatype(C, $E),$N}})
        end
    elseif isconcretetype(T)
        fnames = fieldnames(T)
        ftypes = Expr(:curly, :Tuple, [:(metatype(C, $F)) for F in T.types]...)
        result = :(Meta{metadatatype(C, T),NamedTuple{$fnames,$ftypes}})
    else
        # This commented-out code returns a more specific result, but it's a bit rough on
        # the compiler; to actually compute subtypes correctly here, we'd have to re-expand
        # this generated function for every new world age.
        # metasubtypes = [:(metatype(C, $S)) for S in subtypes(T)]
        # result = :(Union{$(metasubtypes...)})
        result = :(Meta)
    end
    return quote
        $(Expr(:meta, :inline))
        $result
    end
end

# @generated function initmetanode(::C, ::V, metadata::D) where {C<:Context,V,D}
#     if V <: Array
#         return quote
#             $(Expr(:meta, :inline))
#             MetaNode{metatype(C, V)}(metadata, similar(value, metanodetype(C, eltype(V))))
#         end
#     elseif fieldcount(V) == 0
#         return quote
#             $(Expr(:meta, :inline))
#             MetaNode{metatype(C, V)}(metadata, NamedTuple())
#         end
#     else
#         S = isimmutable(V) ? :Immutable : :Mutable
#         fnames = fieldnames(V)
#         fdatas = map(F -> :($S{metanodetype(C, $F)}(unused)), V.types)
#         fields = Expr(:tuple)
#         for i in 1:fieldcount(V)
#             push!(fields.args, :($(fnames[i]) = $(fdatas[i])))
#         end
#         return quote
#             $(Expr(:meta, :inline))
#             MetaNode{metatype(C, V)}(metadata, $fields)
#         end
#     end
# end

#######
# Box #
#######

#=
Here, `U` is the innermost, "underlying" type of the value being wrapped. This parameter is
precomputed so that Cassette can directly dispatch on it in the signatures generated for
contextual primitives.
=#
struct Box{C,U,V,D,T}
    context::C
    value::V
    meta::Meta{D,T}
    function Box(context::C, value::V, meta::Meta{D,T}) where {C<:Context,V,D,T}
        U = _underlying_type(V)
        return new{C,U,V,D,T}(context, value, meta)
    end
end

# function Box(context::C, value, metadata = unused) where {C<:Context}
#     return Box(context, value, initmetanode(context, value, metadata))
# end

@inline _underlying_type(::Type{V}) where {V} = V
@inline _underlying_type(::Type{<:Box{<:Context,U}}) where {U} = U

@inline isboxed(::Context, ::Any) = false
@inline isboxed(::C, ::Box{C}) where {C<:Context} = true

@inline isboxedtype(::Type{C}, ::DataType) where {C<:Context} = false
@inline isboxedtype(::Type{C}, ::Type{<:Box{C}}) where {C<:Context} = true

@inline unbox(::Context, x) = x
@inline unbox(::C, x::Box{C}) where {C<:Context} = x.value

@inline unboxtype(::Type{C}, T::DataType) where {C<:Context,X} = T
@inline unboxtype(::Type{C}, ::Type{<:Box{C,U,V}}) where {C<:Context,U,V} = V

@inline metadata(::C, x::Box{C}) where {C<:Context} = x.meta.data

@inline hasmetadata(::Context, ::Any) = false
@inline hasmetadata(ctx::C, x::Box{C}) where {C<:Context} = isa(metadata(ctx, x), Unused)

@generated function unboxcall(ctx::Context, f, args...)
    N = fieldcount(typeof(args))
    return quote
        $(Expr(:meta, :inline))
        $(Expr(:call, :f, [:(unbox(ctx, args[$i])) for i in 1:N]...))
    end
end

######################
# `new`/`Array{T,N}` #
######################

@generated function _new(::Type{T}, args...) where {T}
    return quote
        $(Expr(:meta, :inline))
        $(Expr(:new, T, [:(args[$i]) for i in 1:nfields(args)]...))
    end
end

@generated function _new_box(ctx::C, ::Type{T}, args...) where {C<:Context,T}
    if !(any(arg <: Box{C} for arg in args)) || !(isconcretetype(T)) || (length(args) != fieldcount(T))
        result = :(_new(T, args...))
    else
        unboxed_args = [:(unbox(ctx, args[$i])) for i in 1:length(args)]
        fields = Expr(:tuple)
        fnames = fieldnames(T)
        for i in 1:fieldcount(T)
            arg = args[i]
            fname = fnames[i]
            mtype = :(metatype(C, $(T.types[i])))
            fmeta = arg <: Box{C} ? :(args[$i].meta::$mtype) : :($mtype(unused, unused))
            push!(fields.args, :($fname = $fmeta))
        end
        new_T = :(_new(T, $(unboxed_args...)))
        meta_T = :(metatype(C, T)(unused, $fields))
        # result = :(Box(ctx, $new_T, $meta_T))
        # if isimmutable(T)
        # else
        #     result = quote
        #         new_T = $new_T
        #         meta_T = $meta_T
        #         # MEMORY[ctx, new_T] = meta_T
        #         return Box(ctx, new_T, meta_T)
        #     end
        # end
    end
    return quote
        $(Expr(:meta, :inline))
        $result
    end
end

# #####################
# # struct primitives #
# #####################
#
# @inline _getfield(x::Box, name) = __getfield(x, unbox(x.context, name))
# @inline __getfield(x::Box, name) = Box(x.context, getfield(x.value, name), getfield(x.meta.tree, name)[])
#
# @inline _setfield!(x::Box, name, y) = __setfield!(x, unbox(x.context, name), y)
# @inline __setfield!(x::Box, name, y) = __setfield!(x, name, y, y, unused)
# @inline __setfield!(x::Box{C}, name, y::Box{C}) where {C} = __setfield!(x, name, y, y.value, y.meta)
# @inline function __setfield!(x::Box, name, y, y_value, y_meta)
#     setfield!(x.value, name, y_value)
#     setindex!(getfield(x.meta.tree, name), y_meta)
#     return y
# end
#
# ####################
# # array primitives #
# ####################
#
# @inline _arrayref(check, x::Box, i) = __arrayref(unbox(x.context, check), x, unbox(x.context, i))
# @inline __arrayref(check, x::Box, i) = Box(x.context, arrayref(check, x.value, i), arrayref(check, x.meta.tree, i))
#
# @inline _arrayset(check, x::Box, y, i) = __arrayset(unbox(x.context, check), x, y, unbox(x.context, i))
# @inline __arrayset(check, x::Box, y, i) = __arrayset(check, x, y, unused, i)
# @inline __arrayset(check, x::Box{C}, y::Box{C}, i) where {C} = __arrayset(check, x, y.value, y.meta, i)
# @inline function __arrayset(check, x::Box, y_value, y_meta, i)
#     arrayset(check, x.value, y_value, i)
#     arrayset(check, x.meta, y_meta, i)
#     return x
# end
#
# @inline __growbeg!(x::Box, delta) = # TODO
# @inline __growend!(x::Box, delta) = # TODO
# @inline __growat!(x::Box, i, delta) = # TODO
# @inline __deletebeg!(x::Box, delta) = # TODO
# @inline __deleteend!(x::Box, delta) = # TODO
# @inline __deleteat!(x::Box, i, delta) = # TODO
