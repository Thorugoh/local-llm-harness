const BASE_URL = "http://192.168.1.17:8083";
const MODEL = "qwen3.6-35b-a3b";

const main = async () => {
    const response = await fetch(`${BASE_URL}/v1/chat/completions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
        model: MODEL,
        messages: [{ role: "user", content: "Explain what a mutex is in one sentence." }],
        stream: false,
    }),
    });

    const data = await response.json();
    console.log(data.choices[0].message.content);
}

main();