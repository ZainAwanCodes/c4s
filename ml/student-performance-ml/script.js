document.getElementById('predict-form').addEventListener('submit', function(e) {
  e.preventDefault();
  
  const btn = document.querySelector('.btn-predict');
  const btnText = btn.querySelector('span');
  const loader = btn.querySelector('.loader');
  const resultBox = document.getElementById('result-container');
  const predictionText = document.getElementById('prediction-text');
  const predictionDesc = document.getElementById('prediction-desc');
  
  // Hide previous result
  resultBox.classList.add('hidden');
  resultBox.classList.remove('pass', 'fail');
  
  // Show loading animation
  btnText.style.opacity = '0';
  loader.classList.remove('hidden');
  btn.style.pointerEvents = 'none';

  // Get values from form
  const inputs = {
    studytime: parseInt(document.getElementById('studytime').value),
    failures: parseInt(document.getElementById('failures').value),
    absences: parseInt(document.getElementById('absences').value),
    goout: parseInt(document.getElementById('goout').value),
    health: parseInt(document.getElementById('health').value)
  };

  // Simulate server/model delay
  setTimeout(() => {
    // Basic heuristic to mock the logistic regression model
    // High study time, low failures, low absences generally lead to Pass
    let score = 0;
    score += (inputs.studytime * 2);
    score -= (inputs.failures * 3);
    score -= (inputs.absences * 0.2);
    score -= (inputs.goout * 0.5);
    score += (inputs.health * 0.2);

    const isPass = score > 1; // Arbitrary threshold for mockup

    // Update UI
    btnText.style.opacity = '1';
    loader.classList.add('hidden');
    btn.style.pointerEvents = 'auto';

    resultBox.classList.remove('hidden');
    
    if (isPass) {
      resultBox.classList.add('pass');
      predictionText.textContent = 'Pass';
      predictionDesc.textContent = 'Based on the metrics, this student is likely to succeed.';
    } else {
      resultBox.classList.add('fail');
      predictionText.textContent = 'Fail';
      predictionDesc.textContent = 'Warning: This student is at high risk of failing.';
    }
    
    // Small pop animation
    resultBox.style.transform = 'scale(1.05)';
    setTimeout(() => {
      resultBox.style.transform = 'scale(1)';
    }, 200);

  }, 1200);
});
