using Pkg

# Define the list of required packages
packages = ["CSV", "DataFrames", "Statistics", "MLJ", "CategoricalArrays", "XGBoost", "JLD2", "MLJXGBoostInterface", "ScientificTypes", "StatsBase"]

# Get the list of installed packages
installed_packages = keys(Pkg.dependencies())

# Check and install missing packages
for pkg in packages
    if pkg ∉ installed_packages  # ∉ means "not in" in Julia
        println("Installing $pkg...")
        Pkg.add(pkg)
    end
end

println("All packages are installed and ready to use!")
