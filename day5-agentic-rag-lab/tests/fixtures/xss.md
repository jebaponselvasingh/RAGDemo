# XSS test document

This file checks that uploaded text is always shown as text, never run as HTML.

<img src=x onerror="window.__xss=1"> <script>window.__xss=2</script>

<b onmouseover="window.__xss=3">hover me</b>
