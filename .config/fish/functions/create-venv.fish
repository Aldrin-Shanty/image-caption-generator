function create-venv
    set -l ENV_DIR "$HOME/Documents/Environments"
    
    if test -z "$argv[1]"
        echo "❌ Please specify a name: create-venv <env_name> [python_version]"
        return 1
    end
    
    set -l VENV_NAME "$argv[1]"
    
    # Determine Python version
    if test -n "$argv[2]"
        set -l PYTHON_VERSION "$argv[2]"
    else
        if command -v python3 >/dev/null
            set PYTHON_VERSION "python3"
        else if command -v python >/dev/null
            set PYTHON_VERSION "python"
        else
            echo "❌ No Python installation found!"
            return 1
        end
    end
    
    # Create the virtual environment
    "$PYTHON_VERSION" -m venv "$ENV_DIR/$VENV_NAME"
    echo "✅ Virtual environment '$VENV_NAME' created using $PYTHON_VERSION."
end
