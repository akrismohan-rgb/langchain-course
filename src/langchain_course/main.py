from dotenv import load_dotenv
import os
load_dotenv()

def main() -> None:
    print("Hello from MY first langchain-course!")
    print(os.getenv("OPENAI_API_KEY"))
    print(os.getenv("GOOGLE_API_KEY"))
    
if __name__ == "__main__":
    main()
    print("Goodbye from MY first langchain-course!")
    
