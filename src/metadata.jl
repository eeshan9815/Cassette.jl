struct UnusedMeta end

generic_nfields(x::Array) = length(x)
generic_nfields(x) = nfields(x)
generic_getfield(x::Array, i) = x[i]
generic_getfield(x, i) = getfield(x, i)

mutable struct IDGenerator
    counter::UInt
end

function generate_id!(idgen::IDGenerator, value)
    if isimmutable(typeof(value))
        idgen.counter += 1
        return idgen.counter
    else
        return UInt(pointer_from_objref(value))
    end
end

# TODO: Should we store both metadata + nested structure in the MetaMemory, or just the
# nested structure?

struct MetaEntry{M}
    meta::M
    nested::Vector{UInt}
end

struct MetaMemory
    cache::Dict{UInt,MetaEntry}
    idgen::IDGenerator
    MetaMemory() = new(Dict{UInt,MetaEntry}, IDGenerator(UInt(0)))
end

function allocate!(memory::MetaMemory, value, meta; overwrite::Bool=false)
    id = generate_id!(memory.idgen, value)
    if !haskey(memory.cache, id)
        nested = UInt[allocate!(memory, generic_getfield(value, i), UnusedMeta())
                      for i in 1:generic_nfields(value)]
        memory.cache[id] = MetaEntry(meta, nested)
    elseif overwrite
        memory.cache[id] = MetaEntry(meta, memory.cache[id].nested)
    end
    return id
end

struct Tagged{T,V}
    tag::T
    value::V
    id::UInt
end

function tag(context::AbstractContext, value, meta=UnusedMeta(); kwargs...)
    id = allocate!(context.memory, value, meta; kwargs...)
    return Tagged(context.tag, value, id)
end
