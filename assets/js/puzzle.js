document.getElementById('puzzle-form').addEventListener('submit', function(event) {
  event.preventDefault();
  var answer = document.getElementById('answer').value.trim();

  // The correct answer is '55834'
  if (answer === '55834') {
    window.location.href = '/solution_page.html';
  } else {
    document.getElementById('feedback').textContent = 'Incorrect Input! Every time you get one wrong, a painting falls off the wall!';
  }
});

