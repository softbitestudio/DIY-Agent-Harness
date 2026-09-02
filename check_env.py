import sys, shutil

def check_setup():
    print("🔍 Checking system requirements...")
    docker = shutil.which("docker")
    python_ver = sys.version_info >= (3, 10)
    
    print(f"  [{'x' if python_ver else ' '}] Python 3.10+")
    print(f"  [{'x' if docker else ' '}] Docker Installed (Required for Level 3)")
    
    if python_ver:
        print("\n🚀 Your ready to Rock N' Roll!!")

if __name__ == "__main__":
    check_setup()
