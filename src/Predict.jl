using XGBoost, DataFrames
using JLD2: @load

# ========== Define Constant Module ==========
module ModelConfig
    const REE_COLUMNS = [
        :LA_ppm, :CE_ppm, :PR_ppm, :ND_ppm, :SM_ppm, :EU_ppm, :GD_ppm, :TB_ppm,
        :DY_ppm, :HO_ppm, :ER_ppm, :TM_ppm, :YB_ppm, :LU_ppm
    ]
end

using .ModelConfig: REE_COLUMNS

# ---------- Load Parameters ----------
@load "normalization_params.jld2" col_means col_stds

# ---------- Load Models ----------
function safe_load_model(path)
    isfile(path) || error("Model file does not exist: $path")
    filesize(path) > 0 || error("Empty model file: $path")
    try
        return XGBoost.load(XGBoost.Booster, path)
    catch e
        @error "Model loading failed" exception=(e, catch_backtrace())
        rethrow(e)
    end
end

boosters = (
    material = safe_load_model("material.model"),
    type = safe_load_model("type.model"),
    composition = safe_load_model("composition.model"),
    rock = safe_load_model("rock.model")
)

# ---------- Category Validation ----------
@load "categories.jld2" material_levels type_levels composition_levels rock_levels

function validate_features(columns, expected)
    mismatch = setdiff(columns, expected)
    isempty(mismatch) || error("Feature mismatch, missing: $mismatch")
end

# Create New Sample
new_sample = DataFrame(
    [col => 0.0 for col in REE_COLUMNS]...  # Initialize all features to 0.0
)

# Please fill in the actual data HERE! (Sample data for demonstration)
sample_values = Dict(
    :LA_ppm => 35.5, :CE_ppm => 83, :PR_ppm => 9.0, :ND_ppm => 32.1,
    :SM_ppm => 6, :EU_ppm => 0.29, :GD_ppm => 8.15, :TB_ppm => 0.85,
    :DY_ppm => 8.8, :HO_ppm => 1.91, :ER_ppm => 5.8, :TM_ppm => 0.79,
    :YB_ppm => 3, :LU_ppm => 0.38
)

for (k,v) in sample_values
    new_sample[!, k] .= v
end

# Dimension correction and standardization
function prepare_input(data, means, stds)
    
    validate_features(names(data), string.(REE_COLUMNS))
    
    matrix = Matrix(data[!, REE_COLUMNS])
    
    standardized = (matrix .- means') ./ stds'
    
    DMatrix(standardized, feature_names=string.(REE_COLUMNS))
end

dmat = prepare_input(new_sample, col_means, col_stds)

# ---------- Prediction Function ----------
function safe_predict(booster, dmat, levels)
    try
        proba = XGBoost.predict(booster, dmat)
        
        proba = ndims(proba) == 1 ? reshape(proba, 1, :) : proba
        
        @assert size(proba, 1) == 1 "There should be only one sample"
        @assert size(proba, 2) == length(levels) "Mismatch in the number of categories"
        
        # Obtain Prediction Results
        max_idx = argmax(proba, dims=2)[1][2]
        max_prob = proba[argmax(proba)]
        
        # Results Validation
        @assert 1 <= max_idx <= length(levels) "Invalid category index"
        @assert 0 <= max_prob <= 1.0 "Invalid confidence level"
        
        (label=levels[max_idx], confidence=max_prob)
    catch e
        @error "Prediction failed" exception=(e, catch_backtrace())
        rethrow(e)
    end
end

# ---------- Execute Prediction ----------
results = Dict(
    :material => safe_predict(boosters[:material], dmat, material_levels),
    :type => safe_predict(boosters[:type], dmat, type_levels),
    :composition => safe_predict(boosters[:composition], dmat, composition_levels),
    :rock => safe_predict(boosters[:rock], dmat, rock_levels)
)

# ---------- Show Results ----------
function format_result(res)
    """
    $(res.label) (Confidence Level: $(round(res.confidence * 100, digits=1))%)
    """
end

println("""
🔍 Prediction Results:
   Material:    $(format_result(results[:material]))
   Type:        $(format_result(results[:type]))
   Composition: $(format_result(results[:composition]))
   Rock:        $(format_result(results[:rock]))
""")
