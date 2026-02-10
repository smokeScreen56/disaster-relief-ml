document.getElementById("disasterForm").addEventListener("submit", async function (e) {
    e.preventDefault();

    const data = {
        severity_score: parseFloat(document.getElementById("severity_score").value),
        total_affected: parseInt(document.getElementById("total_affected").value),
        infrastructure_damage: parseFloat(document.getElementById("infrastructure_damage").value),
        area_affected: parseFloat(document.getElementById("area_affected").value)
    };

    const response = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify(data)
    });

    const result = await response.json();

    document.getElementById("result").classList.remove("hidden");

    const prioritySpan = document.getElementById("priority");
    prioritySpan.innerText = result.priority;
    prioritySpan.className = result.priority;

    document.getElementById("confidence").innerText = result.confidence;

    const reasonsList = document.getElementById("reasons");
    reasonsList.innerHTML = "";
    result.reasons.forEach(r => {
        const li = document.createElement("li");
        li.innerText = r;
        reasonsList.appendChild(li);
    });
});
