📄 Software Requirements Specification (SRS)
Project: Fine-Tuned Caption Generation System (Phi-2 + QLoRA 4-bit)
1. Introduction
1.1 Purpose

This system is designed to generate English social media captions and hashtags using a fine-tuned Phi-2 language model with QLoRA (4-bit quantization).

It takes varied user inputs and produces structured outputs:

Captions: 30–40 words
Hashtags: exactly 5
1.2 Scope

The system will:

Fine-tune Microsoft Phi-2 (2.7B)
Use QLoRA 4-bit quantization
Generate captions for:
Instagram posts
Facebook posts
Support multiple input styles
Ensure strict output format consistency
Output language: English only
2. System Overview

The system includes:

Data Collection Module
Data Preprocessing Module
Fine-Tuning Module (Phi-2 + QLoRA 4-bit)
Inference Module (Caption Generator)
3. System Architecture
Raw Social Media Data (JSON)
        ↓
Data Cleaning & Processing
        ↓
Instruction Dataset (JSONL)
        ↓
Phi-2 + QLoRA 4-bit Fine-Tuning
        ↓
Fine-Tuned Model (Adapters)
        ↓
Caption Generation System
        ↓
User Input → Caption + Hashtags Output
4. Functional Requirements
4.1 Data Input Support

The system shall accept multiple input types:

A. Keyword Input
“gym motivation”
“fat loss journey”
“fitness transformation”
B. Instruction-based Input
“write a caption for Instagram for gym motivation”
“create Facebook post for fat loss journey”
“generate caption for fitness transformation”
C. Natural Language Input
“give me a caption for my gym progress”
“I need a post about my fat loss journey”
“something motivational for fitness transformation”
D. Mixed / Real-world Input
“insta caption gym motivation”
“fb post for workout discipline”
“caption for fat loss transformation please”
4.2 Platform Awareness

The system shall detect intent for:

Instagram caption style
Facebook post style

If not specified, the system should still generate a valid social media caption.

4.3 Dataset Preparation

Dataset must be JSONL format:

{
  "instruction": "Write a social media caption with hashtags.",
  "input": "Facebook post for fat loss journey",
  "output": "consistent effort and discipline shape your fat loss journey stay focused on your health goals and trust the process because progress takes time #fitness #fatloss #motivation #health #discipline"
}
4.4 Fine-Tuning Requirements
Base model: Microsoft Phi-2 (2.7B)
Method: QLoRA (4-bit NF4 quantization)
Framework:
Hugging Face Transformers
PEFT
bitsandbytes
TRL
Training setup:
Optimized for 4GB GPU
Batch size: 1–4
Gradient accumulation enabled
Instruction-based fine-tuning
4.5 Output Requirements

The system must always generate:

Caption length: 30–40 words
Hashtags: exactly 5
Language: English only
Tone: motivational / fitness / lifestyle
Platform-aware formatting (Instagram / Facebook)
4.6 Inference System
Input: Any user prompt (keyword / sentence / instruction)
Output:
Caption
5 hashtags
Response time: < 5 seconds
Must generalize to unseen inputs
5. Non-Functional Requirements
5.1 Performance
Optimized using 4-bit QLoRA
Runs on low-end GPU (4GB VRAM)
Fast inference
5.2 Reliability
95%+ valid structured outputs
No missing hashtags
No incomplete sentences
5.3 Generalization Requirement

The system must handle:

Keywords → “gym motivation”
Instructions → “write Instagram caption for fat loss journey”
Natural language → “I need a caption for my transformation”
Mixed input → “fb post gym discipline please”
5.4 Scalability
Supports 5K–50K+ dataset samples
Supports multiple fitness/lifestyle topics
5.5 Maintainability
Modular dataset pipeline
Easy retraining
Easy model updates
5.6 Portability
Runs on:
Local machine
Google Colab
Cloud GPU
6. Software Requirements
Python 3.10+
transformers
datasets
peft
bitsandbytes
torch
accelerate
trl
7. Hardware Requirements
Minimum
CPU: i5 / Ryzen 5
RAM: 8 GB
GPU: 4 GB VRAM
Recommended
RAM: 16 GB
GPU: 8–16 GB VRAM
8. Dataset Requirements
Format: JSONL
Size: 5,000–50,000 samples
Must include:
Instagram prompts
Facebook prompts
keyword inputs
natural language inputs
mixed inputs
9. Constraints
Caption length: 30–40 words only
Exactly 5 hashtags required
Must be English only
No irrelevant or unsafe outputs
Must strictly follow instruction format
10. Assumptions
Phi-2 model is available via Hugging Face
QLoRA 4-bit training is used
Dataset is clean and balanced
User inputs are flexible and unpredictable
11. Conclusion

The Fine-Tuned Caption Generation System (Phi-2 + QLoRA 4-bit) is a lightweight and efficient AI system designed for generating structured English social media captions for Instagram and Facebook. It supports diverse user inputs and ensures consistent, high-quality outputs optimized for real-world usage.