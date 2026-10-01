from pathlib import Path

root = Path(__file__).resolve().parent

snapshot = root / "sdk/src/fisheries/autonomy/persistence/snapshot.cpp"
text = snapshot.read_text()
old = (
    "    for (const unsigned char byte : value) {\n"
    "        output.push_back(digits[byte >> 4U]);"
)
new = (
    "    for (const char raw_byte : value) {\n"
    "        const auto byte = static_cast<unsigned char>(raw_byte);\n"
    "        output.push_back(digits[byte >> 4U]);"
)
if old in text:
    snapshot.write_text(text.replace(old, new))

fixture = root / "sdk/tests/maritime_ai_cp1/cp1_fixture.hpp"
text = fixture.read_text().replace(
    '{"ground.alpha", 0.0, 0.0, 1.0, 60.0, 0.08, 4.0, true}',
    '{"ground.alpha", 0.0, 0.0, 1.0, 70.0, 0.08, 4.0, true}',
)
fixture.write_text(text)
