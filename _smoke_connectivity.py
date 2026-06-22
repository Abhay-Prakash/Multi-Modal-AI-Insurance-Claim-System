import sys, os, json
sys.path.insert(0, '.')

try:
    from dotenv import load_dotenv
    load_dotenv('.env', override=False)
    load_dotenv('code/.env', override=False)
except ImportError:
    pass

from config import GEMINI_API_KEY, MODEL_NAME

if not GEMINI_API_KEY:
    print('ERROR: GEMINI_API_KEY not set.')
    sys.exit(1)

print(f'GEMINI_API_KEY: ...{GEMINI_API_KEY[-6:]}')
print(f'MODEL: {MODEL_NAME}')

from google import genai
from google.genai import types

client = genai.Client(api_key=GEMINI_API_KEY)

# TEST 1: Text round-trip
print('\n[TEST 1] Text round-trip...')
resp = client.models.generate_content(
    model=MODEL_NAME,
    contents='Reply with exactly: PONG',
    config=types.GenerateContentConfig(temperature=0.0)
)
print(f'  Response: {resp.text.strip()[:60]}')
assert resp.text.strip(), 'Empty response'
print('  PASS')

# TEST 2: JSON mode
print('\n[TEST 2] JSON structured output...')
resp2 = client.models.generate_content(
    model=MODEL_NAME,
    contents='Return a JSON object with key ok set to true.',
    config=types.GenerateContentConfig(
        temperature=0.0,
        response_mime_type='application/json',
    )
)
parsed = json.loads(resp2.text)
print(f'  Parsed: {parsed}')
assert isinstance(parsed, dict), 'Not a dict'
print('  PASS')

# TEST 3: System instruction
print('\n[TEST 3] System instruction...')
resp3 = client.models.generate_content(
    model=MODEL_NAME,
    contents=types.Content(
        role='user',
        parts=[types.Part(text='What is your role?')]
    ),
    config=types.GenerateContentConfig(
        system_instruction='You are a forensic evidence examiner. Say only FORENSIC_READY.',
        temperature=0.0,
    )
)
print(f'  Response: {resp3.text.strip()[:80]}')
print('  PASS')

# TEST 4: Image input
print('\n[TEST 4] Image input...')
from pathlib import Path
dataset = Path('..') / 'dataset'
sample_imgs = list((dataset / 'images' / 'sample').rglob('*.jpg'))[:1]
if not sample_imgs:
    sample_imgs = list((dataset / 'images' / 'test').rglob('*.jpg'))[:1]

test_img_path = None
if sample_imgs:
    test_img_path = sample_imgs[0]
    img_bytes = test_img_path.read_bytes()
    resp4 = client.models.generate_content(
        model=MODEL_NAME,
        contents=types.Content(
            role='user',
            parts=[
                types.Part(text='Describe what you see in 1 sentence.'),
                types.Part(inline_data=types.Blob(mime_type='image/jpeg', data=img_bytes))
            ]
        ),
        config=types.GenerateContentConfig(temperature=0.0)
    )
    print(f'  Image: {test_img_path.name}  ({len(img_bytes)//1024}KB)')
    print(f'  Response: {resp4.text.strip()[:120]}')
    print('  PASS')
else:
    print('  SKIP (no images found)')

# TEST 5: JSON + image + system instruction (the full stack)
print('\n[TEST 5] JSON + image + system instruction...')
if test_img_path:
    schema_prompt = (
        'Describe the main object. '
        'Return JSON: {"object_seen": "string", "damage_visible": true_or_false}'
    )
    resp5 = client.models.generate_content(
        model=MODEL_NAME,
        contents=types.Content(
            role='user',
            parts=[
                types.Part(text=schema_prompt),
                types.Part(inline_data=types.Blob(mime_type='image/jpeg', data=img_bytes))
            ]
        ),
        config=types.GenerateContentConfig(
            system_instruction='You are a forensic examiner. Describe only what you see.',
            temperature=0.0,
            response_mime_type='application/json',
        )
    )
    parsed5 = json.loads(resp5.text)
    print(f'  JSON output: {parsed5}')
    print('  PASS')

print('\n=== ALL CONNECTIVITY TESTS PASSED ===')
print(f'SDK: google-genai, model: {MODEL_NAME}')
