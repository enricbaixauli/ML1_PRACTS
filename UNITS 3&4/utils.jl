# Auxiliary file with functions from previous units

function categorical_encoding(vec::AbstractVector)
    classes = unique(vec)
    n_classes = length(classes)

    if n_classes == 2 return vec .== classes[1]
    else
        n = length(vec)
        M = falses(n, n_classes)  

        for i in 1:n
            for j in 1:n_classes
                if vec[i] == classes[j]
                    M[i, j] = true
                end
            end
        end

        return M
    end
end

function buildClassANN(numInputs::Int, topology::AbstractArray{<:Int,1}, numOutputs::Int;
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)))

    ann = Chain()
    numInputsLayer = numInputs

    for (i, numOutputsLayer) in enumerate(topology)
        ann = Chain(ann..., Dense(numInputsLayer, numOutputsLayer, transferFunctions[i]))
        numInputsLayer = numOutputsLayer
    end

    if numOutputs == 1
        ann = Chain(ann..., Dense(numInputsLayer, 1, σ))
    else
        ann = Chain(ann..., Dense(numInputsLayer, numOutputs, identity), softmax)
    end

    return ann
end           
