module.exports = {
	apps: [
		{
			name: "reluray-api",
			script: "app.py",
			interpreter: "/home/isaac/reluray/backend/venv/bin/python",
			cwd: "/home/isaac/reluray/backend",
			env: {
				PORT: 5001,
				MODEL_VERSION: "1.0.0"
			}
		},
		{
			name: "reluray-web",
			script: "npm",
			args: "start -- -p 3001",
			cwd: "/home/isaac/reluray/frontend",
			env: {
				NODE_ENV: "production",
				NEXT_PUBLIC_API_URL: "/api"
			}
		}
	]
};
