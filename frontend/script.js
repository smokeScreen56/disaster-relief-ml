document.getElementById("predictForm").addEventListener("submit", async function(e){

e.preventDefault();

const data = {
deaths: Number(document.getElementById("deaths").value),
injured: Number(document.getElementById("injured").value),
affected: Number(document.getElementById("affected").value),
homeless: Number(document.getElementById("homeless").value),
damage_usd: Number(document.getElementById("damage").value),
area_affected: Number(document.getElementById("area").value)
};

const response = await fetch("http://localhost:8000/predict",{
method:"POST",
headers:{
"Content-Type":"application/json"
},
body: JSON.stringify(data)
});

const result = await response.json();

let output = "<b>Predicted Priority:</b> " + result.priority + "<br><br>";

if(result.confidence){
output += "<b>Confidence:</b> " + (result.confidence * 100).toFixed(2) + "%<br><br>";
}

if(result.top_factors){
output += "<b>Key Factors:</b><br>";

for(const key in result.top_factors){
output += key + " : " + result.top_factors[key].toFixed(3) + "<br>";
}
}

document.getElementById("result").innerHTML = output;

});