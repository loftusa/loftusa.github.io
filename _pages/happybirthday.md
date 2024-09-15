---
permalink: /happybirthday/
title: "Happy Birthday, Aina!"
layout: single
---

Alex and Aina had just moved into their charming new home in Beacon Hill. The brick-lined streets reminded Aina of the historic alleys back in Spain, except with more bumpy bike lanes, while Alex was excited to start his PhD. The house was perfect, but it still needed that personal touch, and possibly some paint and hanging shelves, to feel like home.

Alex's family flew in from Seattle to help them settle in. Boxes were everywhere, and the walls looked bare. Determined to hang their favorite furniture and decorations, they unboxed their tools and the all-important instruction manual. However, hanging furniture turned out to be more complicated than they ever could have expected!

In the midst of the chaos, they found the instruction manual for hanging one of the paintings. It was surprisingly difficult...

The creatively enhanced instruction manual consisted of lines of text; each line contained a specific measurement that was somehow scrambled. On each line, the measurement can be found by combining the **first digit** and the **last digit** (in that order) to form a single two-digit number. The sum of all of these numbers produces the location that the painting should be hung.

(... Upon reflection, Aina and Alex decided that maybe it would be better to just eyeball it, but they were never ones to shy from a challenge!)

For example:

`1abc2 pqr3stu8vwx a1b2c3d4e5f treb7uchet`


In this example, the measurements of these four lines are **12**, **38**, **15**, and **77**. Adding these together produces **142**.

Consider the entire set of instructions. **What is the sum of all of the measurements?**

*Note: Click [here](/files/aina_bday/bday_puzzle_input.txt) to download the puzzle input.*

<!-- Input field -->
<form id="puzzle-form">
  <label for="answer">Enter your answer:</label><br>
  <input type="text" id="answer" name="answer">
  <button type="submit">Submit</button>
</form>

<p id="feedback" style="color:red;"></p>

<!-- JavaScript -->
<script type="text/javascript">
  document.getElementById('puzzle-form').addEventListener('submit', function(event) {
    event.preventDefault();
    var answer = document.getElementById('answer').value.trim();

    // Replace 'CORRECT_ANSWER' with the actual numerical answer
    if (answer === '55834') {
      window.location.href = '/solution_page.html';
    } else {
      document.getElementById('feedback').textContent = 'Incorrect Input! Ha Ha, You Suck!';
    }
  });
</script>

