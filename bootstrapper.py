import sys
import os
import subprocess
import logging

# Set up logging
logging.basicConfig(filename='bootstrapper.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

ENV_DIR = "app_env"

def create_virtual_env():
    logging.info("Checking for virtual environment at {}".format(ENV_DIR))
    
    if not os.path.exists(ENV_DIR):
        logging.info("Virtual environment not found. Creating a new one.")
        
        # Import virtualenv and create a new environment
        import virtualenv
        virtualenv.create_environment(ENV_DIR)

def install_dependencies():
    logging.info("Installing dependencies.")
    
    # Ensure the requirements.txt file is bundled with your application
    requirements_path = "requirements.txt"
    
    # pip executable within the virtual environment
    pip_path = os.path.join(ENV_DIR, 'Scripts', 'pip')
    
    try:
        subprocess.check_call([pip_path, "install", "-r", requirements_path])
        logging.info("Dependencies installed successfully.")
    except Exception as e:
        logging.error("Error installing dependencies: {}".format(e))

def main():
    #try:
    #	create_virtual_env()
    #except Exception as e:
    #	logging.error("An error occurred in the bootstrapper: {}".format(e), exc_info=True)

    try:
        import langchain
    except ImportError:
        logging.warning("Some dependencies are missing. Attempting to install.")
        install_dependencies()
    
    # Now you can run your main application logic.
    # If it's in another file, you can use exec as shown before.
    try:
        with open('app.py', 'r') as file:
            exec(file.read())
        logging.info("Main application executed successfully.")
    except Exception as e:
        logging.error("Error executing main application: {}".format(e))

if __name__ == "__main__":
    logging.info("Bootstrapper started.")
    try:
        main()
        logging.info("Bootstrapper finished.")
    except Exception as e:
        logging.error("An error occurred in the bootstrapper: {}".format(e))