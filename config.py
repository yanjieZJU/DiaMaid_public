import os

# === OpenAI Configuration ===
os.environ["OPENAI_API_KEY"] = "sk-y44HwiOqlzdvdMwiVGmyzpITjGQdkwWS0bJuDMWN2awpHnX7" 
os.environ["OPENAI_BASE_URL"] = "https://poloai.top/v1"

# === Neo4j Configuration ===
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "BQLjYc4gOpIxTi6o5c-es481vKejEHHBGuxJeEvX9hg"
NEO4J_URL = "neo4j+s://8891a7fc.databases.neo4j.io"

# === API KEY TEST ===
if __name__ == "__main__":
    from openai import OpenAI
    import httpx
    http_client = httpx.Client(proxy=None, trust_env=False)
    client = OpenAI(http_client=http_client)
    
    print(client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "How are you?"}],
        temperature=0.2,
    ).choices[0].message.content)