# Electricity Maps Forecasting

This project aims to predict the carbon intensity of electricity on a given grid using historical data.

## Project Structure

- **data/**: Contains the dataset and preprocessed data files.
- **models/**: Contains the model definition and training scripts.
- **venv/**: Virtual environment directory.
- **features.py**: Defines the selected features.
- **functions.py**: Contains utility functions.
- **main.py**: Main entry point of the application.
- **preprocessing.py**: Handles data preprocessing.
- **requirements.txt**: Lists the dependencies.
- **.gitignore**: Specifies files to be ignored by Git.

## Setup Instructions

1. **Create Virtual Environment**:
    ```sh
    python -m venv venv
    ```

2. **Activate Virtual Environment**:
    - On Windows:
        ```sh
        venv\Scripts\activate
        ```
    - On macOS/Linux:
        ```sh
        source venv/bin/activate
        ```

3. **Install Dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

## Running the Project

1. **Preprocess the Data**:
    ```sh
    python preprocessing.py
    ```

2. **Train and Evaluate the Model**:
    ```sh
    python models/model.py
    ```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.