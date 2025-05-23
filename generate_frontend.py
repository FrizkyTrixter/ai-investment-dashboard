import os

# Define the frontend directory to create
frontend_dir = "frontend_web"
os.makedirs(frontend_dir, exist_ok=True)

# File contents
files = {
    "index.html": """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Backtest Dashboard</title>
    <link rel="stylesheet" href="styles.css">
</head>
<body>
    <div class="container">
        <h1>Run Backtest</h1>
        <form id="configForm">
            <label for="n_trees"># of Trees:</label>
            <input type="number" id="n_trees" name="n_trees" value="100"><br>
            <label for="horizon">Prediction Horizon:</label>
            <input type="number" id="horizon" name="horizon" value="100"><br>
            <label for="rise_thr">Rise Threshold:</label>
            <input type="number" step="0.01" id="rise_thr" name="rise_thr" value="0.1"><br>
            <button type="submit">Run</button>
        </form>
        <div id="results">
            <h2>Results</h2>
            <pre id="output"></pre>
        </div>
    </div>
    <script src="app.js"></script>
</body>
</html>
""",

    "styles.css": """
body {
    font-family: Arial, sans-serif;
    background: #f9f9f9;
    padding: 20px;
}
.container {
    max-width: 600px;
    margin: auto;
    padding: 20px;
    background: white;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    border-radius: 10px;
}
input {
    width: 100%;
    padding: 10px;
    margin-bottom: 15px;
}
button {
    padding: 10px 20px;
    background: #007bff;
    color: white;
    border: none;
    border-radius: 5px;
    cursor: pointer;
}
button:hover {
    background: #0056b3;
}
""",

    "app.js": """
document.getElementById("configForm").addEventListener("submit", async function (e) {
    e.preventDefault();
    const data = {
        n_trees: parseInt(document.getElementById("n_trees").value),
        horizon: parseInt(document.getElementById("horizon").value),
        rise_thr: parseFloat(document.getElementById("rise_thr").value)
    };

    const response = await fetch("http://localhost:5000/run", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(data)
    });

    const result = await response.json();
    document.getElementById("output").textContent = JSON.stringify(result, null, 2);
});
"""
}

# Create each file in the frontend_web directory
for filename, content in files.items():
    with open(os.path.join(frontend_dir, filename), "w") as f:
        f.write(content.strip())

print(f"Frontend scaffold created in ./{frontend_dir}")

