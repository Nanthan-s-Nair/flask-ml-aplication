## Flask-ml-Application

This repository contains a Flask application that uses a trained neural network to recommend suitable fans based on user-provided flow and pressure parameters.

### Features

* **Neural Network Model:** The application utilizes a trained neural network model to predict the required power for a given flow and pressure.
* **User Authentication:** Users can register and login to access the fan recommendation tool.
* **Fan Recommendation:** The application provides a comprehensive recommendation for the most suitable fan, including:
    * Rated Power
    * Other Fan Specifications (e.g., Type, Op. Temp., VFD, Material)
* **User Interface:** A simple and user-friendly web interface allows users to input parameters and receive recommendations.
* **Report Generation:** The application generates a PDF report with detailed information about the recommended fan system.

### Requirements

* Python 3.6+
* Flask
* Pandas
* NumPy
* TensorFlow
* scikit-learn
* Flask-CORS

### Installation

1. Clone the repository: `git clone https://github.com/your-username/fan-recommendation-system.git`
2. Install the required packages: `pip install -r requirements.txt`

### Usage

1. Run the Flask application: `flask run`
2. Access the application in your web browser: `http://127.0.0.1:5000/`
3. Register or login to the application.
4. Enter the desired flow and pressure values.
5. Click "Recommend Fan" to receive the fan recommendations.

### Data

The application uses a CSV file (`data1.csv`) containing historical data on various fans and their corresponding parameters. This data is used to train the neural network model. 

### Contributing

Contributions are welcome! You can contribute by:

* **Improving the model accuracy:** Experiment with different neural network architectures or data preprocessing techniques.
* **Adding new features:** Implement additional functionalities like advanced filtering options or visualizations.
* **Fixing bugs:** Report any bugs or issues you encounter.

### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Acknowledgements

This project is inspired by the need for a quick and efficient solution for selecting suitable fans based on specific requirements. The use of a neural network model allows for a data-driven approach to fan recommendations.
