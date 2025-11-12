function countVehicles(videoUrl) {
    const popup = window.open(videoUrl, "Vehicle Detection", "width=800,height=600");

    popup.onbeforeunload = () => {
        fetch(`/count/?video=${videoUrl}`)
            .then(response => response.json())
            .then(data => {
                let resultHTML = "<h3>Vehicle Counts:</h3><ul>";
                for (let key in data) {
                    resultHTML += `<li>${key}: ${data[key]}</li>`;
                }
                resultHTML += "</ul>";
                document.getElementById("results").innerHTML = resultHTML;
            });
    };
}
