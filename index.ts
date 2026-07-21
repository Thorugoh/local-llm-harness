import { chat } from "./client";

const main = async () => {
    const result = await chat([{
        role: "user",
        content: "Explain what a mutex is in one sentence."
    }]); 

    console.log(JSON.stringify(result, null, 2));
    
}

main();