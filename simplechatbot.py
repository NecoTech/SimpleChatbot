import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class SimpleChatbot:
    def __init__(self, model_name="gpt2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def get_response(self, prompt, max_length=100):
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        
        output = self.model.generate(
            input_ids, 
            max_length=max_length, 
            num_return_sequences=1, 
            no_repeat_ngram_size=2, 
            top_k=50, 
            top_p=0.95, 
            temperature=0.7
        )
        
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return response

def main():
    chatbot = SimpleChatbot()
    print("Simple AI Chatbot (type 'quit' to exit)")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break
        
        response = chatbot.get_response(user_input)
        print(f"AI: {response}")

if __name__ == "__main__":
    main()
