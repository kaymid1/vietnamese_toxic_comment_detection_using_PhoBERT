Key notes before you run
1. One manual step in build.yml:
Replace <your-github-username> in the env: block with your actual GitHub username.

2. Frontend lockfile — add this for faster CI:

Then un-comment cache: 'npm' in the CI workflow and switch to npm ci.

3. models/ directory must exist locally:

4. First docker compose up --build will be slow (~15–20 min) because torch==2.9.1 CPU wheel is ~800 MB. After that, layer cache makes rebuilds fast (2–3 min for code-only changes).

5. Every TODO(gpu-upgrade): comment marks exactly what to change when you switch to a GPU VM — there are 5 spots total across the files.

