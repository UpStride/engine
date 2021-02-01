pipeline {
    agent {
        label 'azure-gpu'
    }
    environment {
        SLACK_WEBHOOK = 'https://hooks.slack.com/services/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'
        REGISTRY_PROD = 'registryupstrideprod.azurecr.io'
        REGISTRY_DEV = 'registryupstridedev.azurecr.io'
        REPO = 'upstride'
        GIT_REPO = 'upstride_python'
        BUILD_TAG = "py"
        BUILD_VERSION = "2.1.0"
    }
    stages {
        stage('setup') {
            steps {
                script {
                    header()
                    info("Starting the pipeline")
                    env.BUILD_VERSION = readFile("version").trim()
                    env.BUILD_DEV = "${REGISTRY_DEV}/${REPO}:${BUILD_TAG}-${BUILD_VERSION}"
                    env.BUILD_PROD = "${REGISTRY_PROD}/${REPO}:${BUILD_TAG}-${BUILD_VERSION}"
                    env.DOCKER_AGENT = "${REGISTRY_DEV}/ops:azure-cloud"
                }
            }
        }
         stage('build docker image') {
            steps {
                script {
                    docker.withRegistry("https://${REGISTRY_DEV}",'registry-dev'){
                        sh("""docker build . -f dockerfile -t $BUILD_DEV """)
                    }
                }
            }
            post {
                success {
                    info('built successful')
                }
                failure {
                    error("stage <build docker> failed")
                }
            }
        }
        stage('smoke tests') {
            options {
                timeout(time: 300, unit: "SECONDS")
            }
            steps {
                script {
                    docker.withRegistry("https://${REGISTRY_DEV}",'registry-dev'){
                        docker.image(env.BUILD_DEV).inside("--gpus all"){
                            tests = ['test.py']
                            for (int i = 0; i < tests.size(); i++) {
                                sh("""python3 ${tests[i]}""")
                            }
                        }
                    }
                }
            }
            post {
                success {
                    info('tests cleared')
                }
                failure {
                    error("stage <smoke tests> failed")
                }
            }
        }
        stage('promote image to dev') {
            when { not { branch 'master' } }
            steps {
                script {
                    docker.withRegistry("https://${REGISTRY_DEV}",'registry-dev'){
                        sh("""docker push $BUILD_DEV """)
                    }
                }
            }
            post {
                success {
                    info("image promoted to dev \n- image: $BUILD_PROD")
                }
                failure {
                    error("stage <promote image to dev> failed")
                }
            }
        }
        stage('promote image to staging') {
            when {  branch 'master'  }
            steps {
                script {
                    docker.withRegistry("https://${REGISTRY_PROD}",'registry-prod'){
                        sh("""docker tag $BUILD_DEV $BUILD_PROD """)
                        sh("""docker push $BUILD_PROD """)
                    }
                }
            }
            post {
                success {
                    info("image promoted to staging \n- image: $BUILD_PROD")
                }
                failure {
                    error("stage <promote image to staging> failed")
                }
            }
        }
    }
    post {
        success {
            info("pipeline **SUCCESS**")
        }
        failure {
            error("pipeline **FAILURE**")
        }
        always {
            info("logs :${BUILD_URL}consoleText")
            slack()
        }
    }
}

def header(){
    env.SLACK_HEADER = "[META]\n-repo <"+env.GIT_REPO+">\n- push on branch <"+env.GIT_BRANCH+">\n- author <"+env.GIT_COMMITTER_NAME+">"
    env.SLACK_MESSAGE = ''
}

def slack(){
    sh 'echo into slack :: - exiting -'
    DATA = '\'{"text":"'+env.SLACK_HEADER+env.SLACK_MESSAGE+'"}\''
    sh """
    curl -X POST -H 'Content-type: application/json' --data ${DATA} --url $SLACK_WEBHOOK
    """
}

def info(String body){
    env.SLACK_MESSAGE = env.SLACK_MESSAGE+'\n[INFO] '+body.toString()
}

def error(String body){
    env.SLACK_MESSAGE = env.SLACK_MESSAGE+'\n[ERROR] '+body.toString()
}

