from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.prompts import PromptTemplate
import torch

def main():
    """
    Main function to set up and run the language model
    """
    # Initialize model and tokenizer
    model_id = "facebook/opt-125m"  # Using a smaller model for faster loading
    
    # Load tokenizer and model without device mapping
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32
    )

    # Move model to appropriate device after loading
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device)

    # Create pipeline with proper parameters
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=100,
        do_sample=True,  # Enable sampling
        temperature=0.7,
        top_p=0.95,
        repetition_penalty=1.15
    )

    # Create LangChain HuggingFacePipeline instance
    local_llm = HuggingFacePipeline(pipeline=pipe)

    # Create a prompt template
    template = """Question: {question}

    Answer: Let me think about this step by step:"""

    prompt = PromptTemplate(template=template, input_variables=["question"])

    # Use the new LangChain syntax
    chain = prompt | local_llm

    # Example usage
    question = "What are the three states of matter?"
    response = chain.invoke({"question": question})
    print(f"Question: {question}")
    print(f"Response: {response}")

if __name__ == "__main__":
    main()