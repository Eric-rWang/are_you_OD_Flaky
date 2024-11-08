# Are You OD Flaky? Flaky Test Detection and Fixing Using CodeBERT
Using pre-trained and fine-tuned LLMS for automatic OD flaky test detection and repair. 

## Project Overview
This project focuses on detecting and fixing order-dependent flaky tests in Python code using CodeBERT. The goal is to leverage machine learning models to identify flaky tests and suggest fixes to stabilize them, enhancing the reliability of test suites.

### Key Objectives:
- **Classification of Flaky Tests**: Use a fine-tuned CodeBERT model to classify Python test functions as flaky or non-flaky.
- **Fixing Flaky Tests**: Implement a sequence-to-sequence model with CodeBERT to generate modifications for flaky test functions, making them stable.

## Data Source
The flaky tests used in this project are sourced from [iDFlakies](https://sites.google.com/view/flakytestdataset), which provides a collection of flaky tests. All of the tests used in this project are written in Python.

## Project Structure
- **`data/`**: Contains training and validation datasets, including flaky test examples from iDFlakies.
- **`notebooks/`**: Jupyter notebooks used for exploratory data analysis and model prototyping.
- **`models/`**: Pre-trained and fine-tuned models used for classification and sequence generation.
- **`scripts/`**: Python scripts for data preparation, model training, evaluation, and inference.
- **`results/`**: Contains the results of model training, including metrics and generated fixes for flaky tests.

## Progress So Far
- **Migration to CodeBERT**: Transitioned from using ChatGPT to CodeBERT, a model specifically designed for understanding and generating code.
- **Classification of Flaky Tests**: A classification pipeline has been successfully implemented to detect flaky tests. The model is fine-tuned using labeled Python test functions.
- **Sequence-to-Sequence Model**: Initial work on a sequence-to-sequence model for fixing flaky tests has been implemented, although more work is needed to refine and validate this approach.

## Challenges
- **Data Limitations**: The limited amount of labeled training data has been a significant bottleneck. Efforts are ongoing to expand the dataset to improve the model's performance.
- **Training Complexity**: Fine-tuning the model has involved careful adjustment of hyperparameters to prevent overfitting while achieving generalizability.

## Next Steps
1. **Expand the Dataset**: Collect more labeled examples and explore data augmentation techniques to improve the diversity of the training data.
2. **Refine the Sequence-to-Sequence Model**: Continue fine-tuning the sequence-to-sequence model to effectively generate fixes for flaky tests.
3. **Model Evaluation**: Introduce additional evaluation metrics and validation procedures to better understand and improve model performance.

## Getting Started
### Prerequisites
- Python 3.7 or higher
- PyTorch
- Transformers library by Hugging Face
- Jupyter Notebook (optional, for running and modifying notebooks)

### Installation
1. Clone the repository:
   ```sh
   git clone <repository-url>
   cd flaky-test-detection
   ```
2. Create a virtual environment and install the requirements:
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   pip install -r requirements.txt
   ```

### Usage
- **Training the Classification Model**: Use `scripts/train_classification.py` to train the classification model on the provided dataset.
- **Generating Fixes**: Use `scripts/train_seq2seq.py` to fine-tune the sequence-to-sequence model for generating fixes for flaky tests.
- **Evaluation**: The evaluation scripts are located in `scripts/evaluate.py` and can be used to assess the performance of both models.

## Contributing
Contributions are welcome! Please feel free to submit a pull request or create an issue if you encounter any problems or have suggestions for improvements.

## License
This project is licensed under the MIT License.

## Acknowledgments
- Thanks to [iDFlakies](https://sites.google.com/view/flakytestdataset) for providing the flaky test dataset.
- Special thanks to the Hugging Face team for the Transformers library and to Microsoft for CodeBERT.

