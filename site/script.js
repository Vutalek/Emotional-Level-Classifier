async function api_request(heart_rate, skin_conductance, eeg, activity_level) {
    const response = await fetch("http:/127.0.0.1:7070/predict", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Auth": "1234"
        },
        body: JSON.stringify({
            heartRate: parseFloat(heart_rate),
            skinConductance: parseFloat(skin_conductance),
            eeg: parseFloat(eeg),
            temperature: null,
            pupilDiameter: null,
            smileIntensity: null,
            frownIntensity: null,
            cortisolLevel: null,
            activityLevel: parseFloat(activity_level),
            ambientNoiseLevel: null,
            lightingLevel: null
        })
    });
    if (response.ok === true) {
        const prediction = await response.json()
        document.getElementById("engagement_level").innerHTML = "<b>Engagement level:</b> " + prediction.engagementLevel
        document.getElementById("emotional_state").innerHTML = "<b>Emotional State:</b> " + prediction.emotionalState
    }
    else {
        const error = await response.json()
        console.log(error.message)
    }
}

document.getElementById("send_btn").addEventListener("click", async () => {
    const heartRate = document.getElementById("heart_rate").value;
    const skinCondutance = document.getElementById("skin_conductance").value;
    const eeg = document.getElementById("eeg").value;
    const activityLevel = document.getElementById("activity_level").value;

    await api_request(heartRate, skinCondutance, eeg, activityLevel)
});