# CS4824-Final-Project
This repository contains the code for the "Explainable LLM Hallucination Detection" final project for the CE4824 course.

## Project Description
The goal of this project is to develop a system to detect LLM hallucinations using various methods and provide an explanation for each ‘fake’ classification using an LLM.

The dataset that will be used for ground truth and validation purposes is the HiTZ/This-is-not-a-dataset from Hugging Face.

First, the generator and detector will be set up as LLM agents, each with their own agent prompts.

Multiple methods will be used in order to perform binary classification on the outputs:
* Logistic Regression as the baseline
* Neural Network (NN) with TF‑IDF​ (Term Frequency – Inverse Document Frequency) input
* NN with Sentence Transformer (all-MiniLM-L6-v2) input [8]
* LLM detector

For each of these methods, the detector LLM will also be asked to provide an explanation. For all non LLM classifications, the LLM will be provided the classification decided by the classifier and will generate an explanation based on that. For the LLM detector, the classification and explanation are generated together.  

The number of correct and incorrect classifications of each type is then recorded during the validation stage using inputs that the classifier has not been trained on, and the data is used to generate a confusion matrix.