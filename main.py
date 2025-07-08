"""Main entry point for the Advanced RAG Chatbot Framework."""

import sys
import os
import subprocess
from pathlib import Path

def check_requirements():
    """Check that all requirements are met."""
    project_root = Path(__file__).parent
    
    # Check for data directory
    data_dir = project_root / "data"
    if not data_dir.exists():
        print("Warning: data/ directory not found. Creating it...")
        data_dir.mkdir(exist_ok=True)
    
    # Check for environment file
    env_file = project_root / ".env"
    if not env_file.exists():
        env_example = project_root / ".env.example"
        if env_example.exists():
            print("Warning: .env file not found")
            print("Please copy .env.example to .env and configure your settings:")
            print(f"  cp {env_example} {env_file}")
        else:
            print("Warning: No .env file found. Please create one with your configuration:")
            print("  OPENAI_API_KEY=your_api_key_here")
        return False
    
    return True

def run_streamlit():
    """Run the Streamlit application."""
    src_path = Path(__file__).parent / "src"
    app_path = src_path / "app.py"
    
    if not app_path.exists():
        print(f"Error: Application file not found at {app_path}")
        sys.exit(1)
    
    try:
        print("🚀 Starting Advanced RAG Chatbot Framework...")
        print("📖 The application will open in your default web browser")
        print("🛑 Press Ctrl+C to stop the application")
        print("-" * 50)
        
        # Run Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", str(app_path),
            "--server.headless", "true",
            "--server.enableCORS", "false",
            "--server.enableXsrfProtection", "false"
        ])
        
    except KeyboardInterrupt:
        print("\n\n🛑 Application stopped by user")
    except Exception as e:
        print(f"\n❌ Error running application: {e}")
        sys.exit(1)

def main():
    """Main entry point."""
    print("🤖 Advanced RAG Chatbot Framework")
    print("=" * 40)
    
    # Check requirements
    if not check_requirements():
        print("\n⚠️  Please fix the issues above and try again.")
        sys.exit(1)
    
    # Add src to Python path
    src_path = Path(__file__).parent / "src"
    sys.path.insert(0, str(src_path))
    
    # Run the application
    run_streamlit()

if __name__ == "__main__":
    main()
