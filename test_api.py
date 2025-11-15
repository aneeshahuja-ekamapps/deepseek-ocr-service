"""
Test script for DeepSeek-OCR API Service
Usage: python test_api.py <API_URL> [API_KEY]
"""

import sys
import requests
import argparse
from pathlib import Path


def test_health(api_url: str):
    """Test health endpoint"""
    print("\n" + "="*60)
    print("Testing Health Endpoint")
    print("="*60)

    try:
        response = requests.get(f"{api_url}/health", timeout=10)
        response.raise_for_status()

        data = response.json()
        print(f"✅ Status: {data['status']}")
        print(f"✅ Model Loaded: {data['model_loaded']}")
        print(f"✅ Device: {data['device']}")
        print(f"✅ GPU Available: {data['gpu_available']}")
        if data.get('gpu_name'):
            print(f"✅ GPU Name: {data['gpu_name']}")

        return True

    except requests.exceptions.RequestException as e:
        print(f"❌ Health check failed: {str(e)}")
        return False


def test_ocr(api_url: str, file_path: str, api_key: str = None):
    """Test OCR processing endpoint"""
    print("\n" + "="*60)
    print("Testing OCR Processing Endpoint")
    print("="*60)

    if not Path(file_path).exists():
        print(f"❌ File not found: {file_path}")
        return False

    try:
        # Prepare request
        headers = {}
        if api_key:
            headers['X-API-Key'] = api_key

        with open(file_path, 'rb') as f:
            files = {'file': f}

            print(f"📤 Uploading file: {file_path}")
            print(f"⏳ Processing... (this may take 10-30 seconds)")

            response = requests.post(
                f"{api_url}/ocr/process",
                files=files,
                headers=headers,
                timeout=60
            )

        response.raise_for_status()
        data = response.json()

        if data['success']:
            print(f"✅ OCR Success!")
            print(f"\n📄 Extracted Markdown (first 500 chars):")
            print("-" * 60)
            print(data['markdown_text'][:500])
            if len(data['markdown_text']) > 500:
                print(f"\n... ({len(data['markdown_text'])} total characters)")
            print("-" * 60)
            print(f"\n📊 Model Info:")
            print(f"   - Model: {data['model_info'].get('model_name', 'N/A')}")
            print(f"   - Device: {data['model_info'].get('device', 'N/A')}")
            print(f"   - Max Tokens: {data['model_info'].get('max_tokens', 'N/A')}")
            return True
        else:
            print(f"❌ OCR Failed: {data.get('error', 'Unknown error')}")
            return False

    except requests.exceptions.Timeout:
        print("❌ Request timeout (>60s). Server might be overloaded or processing a large file.")
        return False
    except requests.exceptions.RequestException as e:
        print(f"❌ OCR test failed: {str(e)}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Test DeepSeek-OCR API Service',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test health only
  python test_api.py http://your-server-ip

  # Test with a document
  python test_api.py http://your-server-ip --file document.pdf

  # Test with API key
  python test_api.py http://your-server-ip --file document.pdf --api-key your-secret-key
        """
    )

    parser.add_argument(
        'api_url',
        help='API URL (e.g., http://your-server-ip or http://localhost:8000)'
    )
    parser.add_argument(
        '--file', '-f',
        help='Path to document file for OCR testing',
        default=None
    )
    parser.add_argument(
        '--api-key', '-k',
        help='API key for authentication',
        default=None
    )

    args = parser.parse_args()

    # Remove trailing slash from URL
    api_url = args.api_url.rstrip('/')

    print("\n" + "="*60)
    print("DeepSeek-OCR API Service Test")
    print("="*60)
    print(f"API URL: {api_url}")
    if args.api_key:
        print(f"API Key: {'*' * len(args.api_key)}")

    # Test health endpoint
    health_ok = test_health(api_url)

    if not health_ok:
        print("\n❌ Health check failed. Please ensure the service is running.")
        sys.exit(1)

    # Test OCR if file provided
    if args.file:
        ocr_ok = test_ocr(api_url, args.file, args.api_key)

        if not ocr_ok:
            print("\n❌ OCR test failed.")
            sys.exit(1)
    else:
        print("\n💡 Tip: Use --file to test OCR processing")

    print("\n" + "="*60)
    print("✅ All tests passed!")
    print("="*60)
    print("\n📚 API Documentation available at:")
    print(f"   {api_url}/docs")
    print()


if __name__ == "__main__":
    main()
