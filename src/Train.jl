module MainModule
    using CSV, DataFrames, Statistics, MLJ, CategoricalArrays, XGBoost, JLD2
    using MLJXGBoostInterface
    using ScientificTypes
    using StatsBase
    using JLD2: @save, @load  
    export main, REE_COLUMNS
    
    # ========== Define Constants ==========
    const REE_COLUMNS = [
        :LA_ppm, :CE_ppm, :PR_ppm, :ND_ppm, :SM_ppm, :EU_ppm, :GD_ppm, :TB_ppm,
        :DY_ppm, :HO_ppm, :ER_ppm, :TM_ppm, :YB_ppm, :LU_ppm
    ]

    # ========== Data Pre-processing Functions ==========
    function process_numeric!(df, col)
        df[!, col] = map(df[!, col]) do x
            raw_str = replace(string(something(x, "")), r"[<≈≤≥]" => "")
            parsed = tryparse(Float64, raw_str)
            isnothing(parsed) ? 0.0 : parsed
        end
        df[!, col] = convert(Vector{Float64}, coalesce.(df[!, col], 0.0))
    end

    function load_data()
        file_path = "C:\\……your_own_path\\earthchem_download_32495.csv"
        df = CSV.read(file_path, DataFrame; header=6)

        # Select feature columns
        select!(df, [
            :MATERIAL, :TYPE, :COMPOSITION, :ROCK,  
            :SR87_SR86_, :RB87_SR86_,
            REE_COLUMNS...
        ])

        # Processing category columns (converted to strings)
        for col in [:MATERIAL, :TYPE, :COMPOSITION, :ROCK]
            df[!, col] = coalesce.(string.(df[!, col]), "Unknown")
        end

        # Process all numeric columns
        for col in REE_COLUMNS
            process_numeric!(df, col)
        end

        return df
    end

    # ========== Category Label Process ==========
    function safe_categorical(v)
        clean_v = coalesce.(string.(v), "Unknown")
        counts = countmap(clean_v)
        ordered_levels = sort(collect(keys(counts)), by=x->counts[x], rev=true)
        
        cat_v = categorical(clean_v, 
                           levels=ordered_levels,
                           ordered=true,
                           compress=true)
        
        # Validate ScientificTypes
        @assert scitype(cat_v) <: AbstractVector{<:OrderedFactor} "Type Conversion Failure"
        return cat_v, ordered_levels
    end

    # ========== Main ==========
    function main()
        # 1. Load data
        df = load_data()
        @info "Data loading complete, number of samples: $(nrow(df))"

        # 2. Split dataset
        train, test = partition(eachindex(df.ROCK), 0.8, shuffle=true)
        train_df = df[train, :]
        test_df = df[test, :]
        @info "Dataset split complete, training set: $(nrow(train_df)), test set: $(nrow(test_df))"

        # 3. Data Standardization
        col_means = [mean(skipmissing(train_df[:, col])) for col in REE_COLUMNS]
        col_stds = [std(skipmissing(train_df[:, col])) for col in REE_COLUMNS]

        for (i, col) in enumerate(REE_COLUMNS)
            train_df[!, col] = (train_df[!, col] .- col_means[i]) ./ col_stds[i]
            test_df[!, col] = (test_df[!, col] .- col_means[i]) ./ col_stds[i]
        end
        @info "Data standardization complete"

        # 4. Process Category Labels
        y_train_rock, rock_levels = safe_categorical(train_df.ROCK)
        y_train_material, material_levels = safe_categorical(train_df.MATERIAL)
        y_train_type, type_levels = safe_categorical(train_df.TYPE)
        y_train_composition, composition_levels = safe_categorical(train_df.COMPOSITION)

        # 5. Label Validation
        println("\n=== Label Type Validation ===")
        for (name, col) in [
            (:ROCK, y_train_rock),
            (:MATERIAL, y_train_material),
            (:TYPE, y_train_type),
            (:COMPOSITION, y_train_composition)
        ]
            actual_type = scitype(col)
            expected_type = AbstractVector{<:OrderedFactor}
            println("Label $name Type: $actual_type")
            @assert actual_type <: expected_type "Label $name Type Conversion Failure"
        end

        # 6. Save Pre-processing Parameters
        JLD2.@save "normalization_params.jld2" col_means col_stds REE_COLUMNS
        JLD2.@save "categories.jld2" material_levels type_levels composition_levels rock_levels

        # 7. Prepare Training Data
        X_train = Matrix{Float32}(train_df[!, REE_COLUMNS])
        X_test = Matrix{Float32}(test_df[!, REE_COLUMNS])

        # 8. Train Model
        function train_model(X, y, num_classes::Int)
            X_table = MLJ.table(Matrix{Float32}(X))
            model = XGBoostClassifier(
                objective = num_classes > 2 ? "multi:softprob" : "binary:logistic",
                num_round = 500,
                eta = 0.05,
                max_depth = 6,
                subsample = 0.8,
                colsample_bytree = 0.8,
                tree_method = "hist"
            )
            mach = machine(model, X_table, y; scitype_check_level=0)
            fit!(mach)
            return fitted_params(mach).fitresult[1]
        end
        
        # Train all models
        boosters = [
            ("material.model", train_model(X_train, y_train_material, length(material_levels))),
            ("type.model", train_model(X_train, y_train_type, length(type_levels))),
            ("composition.model", train_model(X_train, y_train_composition, length(composition_levels))),
            ("rock.model", train_model(X_train, y_train_rock, length(rock_levels)))
        ]

        # 9. Save models
        for (filename, booster) in boosters
            XGBoost.save(booster, filename)
        end

        println("\n🎉 All model training completed and saved!")
        println("Saved model files:")
        for (filename, _) in boosters
            println("- ", filename)
        end
    end
end

# Execution
MainModule.main()
