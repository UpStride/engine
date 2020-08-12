pipeline {
    agent { docker { image 'localhost:5000/dtr/azure-cloud' } }

    stages {
        stage('env') {
            steps {
                script {
                    env.SLACK_WEBHOOK = 'https://hooks.slack.com/services/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'
                    env.SLACK_POST = '\'{"text":"integration of branch <'+env.GIT_BRANCH+'>"}\''
                }
            }
        }
        stage('slack') {
            steps {
                script {
					if (env.SLACK_POST == null) {
						throw new Exception("x Missing webhook for notification :: ABORT");
					}
                sh """
                	curl -X POST -H \'Content-type: application/json\' --data $SLACK_POST --url $SLACK_WEBHOOK
                """
				}
			}
		}
	}
}


