const path = require('path');

module.exports = {
	apps: [
		{
			name: "reluray-api",
			script: "app.py",
			interpreter: path.join(__dirname, "backend", "venv", "bin", "python"),
			cwd: path.join(__dirname, "backend"),
			env: {
				PORT: 5001,
				MODEL_VERSION: "1.0.0"
			}
		},
		{
			name: "reluray-web",
			script: "npm",
			args: "start -- -p 3001",
			cwd: path.join(__dirname, "frontend"),
			env: {
				NODE_ENV: "production",
				NEXT_PUBLIC_API_URL: "/api"
			}
		}
	]
};