# Order-Dependent Flaky Test Detection and Repair

This project provides a comprehensive pipeline for detecting and repairing order-dependent (OD) flaky tests in Python. By leveraging machine learning models to identify flaky tests and using ChatGPT for automated code repair, this approach streamlines test maintenance and enhances CI reliability.

## Features

1. **OD Flaky Test Detection**  
   - Fine-tunes a CodeBERT model to classify tests as order-dependent flaky or not.
   - Handles class imbalance via oversampling.
   - Tokenizes and encodes tests using Hugging Face’s `transformers` library for stable training and inference.

2. **Model Fine-Tuning**  
   - Utilizes Hugging Face’s `Trainer` and `TrainingArguments` to simplify hyperparameter tuning and checkpoint management.
   - Allows customization of learning rate, batch size, epochs, and other settings to achieve robust performance.

3. **Automated Test Repair with ChatGPT**  
   - Once flaky tests are identified, ChatGPT is prompted to suggest deterministic fixes.
   - In-context learning provides examples of flaky tests and their repaired counterparts, guiding ChatGPT towards more accurate solutions.
   - Produces code modifications that remove non-deterministic behavior and stabilize tests.

4. **Integration and Extensibility**  
   - Compatible with existing CI/CD pipelines for continuous monitoring and maintenance of test quality.
   - Easily extended to new fix categories, additional language models, or alternative code representations.
   - Scalable approach that can evolve as new models and techniques emerge.

## Typical Workflow

1. **Data Preparation**  
   - Collect Python test methods and label them as flaky or not.
   - Tokenize and encode test code snippets.

2. **Model Training**  
   - Fine-tune a CodeBERT model for binary classification (flaky vs. non-flaky).
   - Evaluate and tune hyperparameters using a validation set.
   - Save the best model checkpoint for future inference.

3. **Inference**  
   - Run the fine-tuned model on new tests to identify OD flaky ones.
   - Output a list of flaky tests along with root cause categories (e.g., environment dependency, global state issues).

4. **Repair Using ChatGPT**  
   - Prompt ChatGPT with the flaky test code and known fix categories.
   - Optionally provide examples of similar flakes and their fixes (in-context learning).
   - Receive and review the repaired test code; verify stability by re-running the tests.

5. **Verification and Deployment**  
   - Validate that the repaired tests pass consistently.
   - Integrate detection and repair into CI pipelines to ensure ongoing test health.

## Benefits

- **Increased Reliability:** Eliminates flaky tests that undermine developers’ trust in test suites.
- **Reduced Maintenance Effort:** Automates both detection and repair, minimizing manual debugging.
- **Faster Feedback Loops:** Stabilized tests lead to more reliable CI and quicker deployments.

## Future Directions

- Experiment with other pre-trained models (e.g., UniXcoder) for improved detection accuracy.
- Expand the set of fix categories and refine prompt engineering for ChatGPT.
- Explore GPT-4 or reinforcement learning techniques for iterative and self-improving test repair.

---

This README provides an overview of the project’s purpose, approach, and usage pattern, enabling contributors and users to quickly understand and adopt the pipeline.
