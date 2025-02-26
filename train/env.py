#!/usr/bin/env python3
"""
Environment variable retrieval utility.
Usage: python env.py VAR_NAME
"""

import sys
import os
import re

# Function to read .env file into a dictionary
def load_env_file(filepath='.env'):
    """
    Read an environment file and return a dictionary of key-value pairs.
    File format should be KEY=VALUE with one pair per line.
    """
    env_vars = {}
    
    try:
        with open(filepath, 'r') as file:
            for line in file:
                line = line.strip()
                
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue
                
                # Match key-value pairs
                match = re.match(r'^([A-Za-z0-9_]+)=(.*)$', line)
                if match:
                    key, value = match.groups()
                    
                    # Remove quotes if present
                    if (value.startswith('"') and value.endswith('"')) or \
                       (value.startswith("'") and value.endswith("'")):
                        value = value[1:-1]
                        
                    env_vars[key] = value
    except FileNotFoundError:
        print(f"Warning: Environment file '{filepath}' not found. Using defaults.")
        
    return env_vars

# Load environment variables from .env file
env_vars = load_env_file()

def get_env_var(var_name):
    """Get the value of an environment variable."""
    # First check if it's in the dictionary
    if var_name in env_vars:
        return env_vars[var_name]
    
    # If not in dictionary, check actual environment variables
    if var_name in os.environ:
        return os.environ[var_name]
    
    return f"Error: Environment variable '{var_name}' not found"

def main():
    """Main function to handle command line arguments."""
    # Display usage if no arguments provided
    if len(sys.argv) < 2:
        print("Usage: python env.py VAR_NAME")
        print("       python env.py --file=path/to/env/file VAR_NAME")
        print("Available variables:", ", ".join(env_vars.keys()))
        return 1
    
    # Check if an alternate env file path is provided
    file_path = '.env'
    var_name = sys.argv[1]
    
    # Check if first argument is specifying an env file path
    if var_name.startswith('--file='):
        file_path = var_name.split('=', 1)[1]
        
        # Make sure there's a second argument for the variable name
        if len(sys.argv) < 3:
            print("Error: No variable name provided.")
            return 1
        
        var_name = sys.argv[2]
        
        # Reload env vars with the specified file
        global env_vars
        env_vars = load_env_file(file_path)
    
    # Get and print the value
    value = get_env_var(var_name)
    print(value)
    return 0

if __name__ == "__main__":
    sys.exit(main())