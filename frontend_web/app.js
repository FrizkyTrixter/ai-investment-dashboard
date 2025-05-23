document.getElementById("configForm").addEventListener("submit", function (e) {
    e.preventDefault();

    const output = document.getElementById("streamOutput");
    output.textContent = "⏳ Running...";

    const data = {
        n_trees: parseInt(document.getElementById("n_trees").value),
        horizon: parseInt(document.getElementById("horizon").value),
        rise_thr: parseFloat(document.getElementById("rise_thr").value)
    };

    const eventSource = new EventSourcePolyfill("/stream", {
        headers: { "Content-Type": "application/json" },
        payload: JSON.stringify(data),
        onmessage: function (event) {
            // ✅ Replace the content instead of appending it
            output.textContent = event.data;
        },
        onerror: function (err) {
            output.textContent = "❌ Stream error occurred\n";
            console.error("Stream error:", err);
        }
    });
});

// Polyfill EventSource for POST streaming using XMLHttpRequest
class EventSourcePolyfill {
    constructor(url, options) {
        const xhr = new XMLHttpRequest();
        xhr.open("POST", url);
        xhr.setRequestHeader("Content-Type", options.headers["Content-Type"]);
        xhr.send(options.payload);

        const stream = new ReadableStream({
            start(controller) {
                xhr.onreadystatechange = () => {
                    if (xhr.readyState === 3 || xhr.readyState === 4) {
                        controller.enqueue(xhr.responseText);
                        if (xhr.readyState === 4) controller.close();
                    }
                };
            }
        });

        const reader = stream.getReader();

        reader.read().then(function pump({ value, done }) {
            if (value) {
                const lines = value.split("\n\n");
                for (const chunk of lines) {
                    if (chunk.startsWith("data: ")) {
                        const message = chunk.replace("data: ", "").trim();
                        if (message && options.onmessage) options.onmessage({ data: message });
                    }
                }
            }
            if (!done) return reader.read().then(pump);
        }).catch(err => {
            if (options.onerror) options.onerror(err);
        });
    }

    close() {}
}
