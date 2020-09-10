import java.util.logging.FileHandler
import java.util.logging.SimpleFormatter
import java.util.logging.LogManager
import jenkins.model.Jenkins

pipeline {
    agent {
        label 'azure-gpu'
        //label 'azure-cpu'
    }
    environment {
        SLACK_WEBHOOK = 'https://hooks.slack.com/services/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'
        REGISTRY_PROD = 'registryupstrideprod.azurecr.io'
        REGISTRY_DEV = 'registryupstridedev.azurecr.io'
        REPO = 'upstride'
    }
    stages {
        stage('setup') {
            steps {
                script {
                    header()
                    info("Starting the pipeline")
                    env.BUILD_TAG = "upstride-python"
                    env.BUILD_VERSION = readFile("version")
                    //env.BUILD_DEV = "${REGISTRY_DEV}/${REPO}:${BUILD_TAG}-${BUILD_VERSION}"
                    env.BUILD_DEV = "${REGISTRY_DEV}/${REPO}/${BUILD_TAG}:${BUILD_VERSION}"
                    env.BUILD_PROD = "${REGISTRY_PROD}/${REPO}:${BUILD_TAG}-${BUILD_VERSION}"
                    env.DOCKER_AGENT = "${REGISTRY_DEV}/ops:azure-cloud"
                    setLogger()
                }

            }
        }
         stage('build docker image') {
            //agent { docker { image "$DOCKER_AGENT" } }
            steps {
                script {
                    docker.withRegistry("https://${REGISTRY_DEV}",'registry-dev'){
                        shell("""docker build . -f dockerfile -t $BUILD_DEV """)
                        info('built successful')
                    }
                }
            }
        }
        stage('smoke tests') {
            options {
                timeout(time: 300, unit: "SECONDS")
            }
            //agent { docker { image "$BUILD_DEV" } }
            //agent { docker { image "tensorflow/tensorflow:2.3.0-gpu" } }
            //agent { docker { image "$DOCKER_AGENT" } }
            steps {
                script {
                    docker.withRegistry("https://${REGISTRY_DEV}",'registry-dev'){
                        //docker.image("${env.BUILD_DEV}").inside { c ->
                            //shell("""pip install .""")
                            shell("""docker build . -f dockerfile -t $BUILD_DEV """)
                            info('built successful')
                            //shell("""docker push $BUILD_DEV """)
                            //info('image promoted to dev')
                            //def build = docker.build("${env.BUILD_DEV}")
                            docker.image(env.BUILD_DEV).inside("--gpus all"){
                            tests = ['test.py', 'test_tf.py', 'test_type1.py','test_type2.py', 'test_type3.py']
                            for (int i = 0; i < tests.size(); i++) {
                                shell("""python3 ${tests[i]}""")
                            }
                            info('tests cleared')
                            }
                            }
                        //}
                }
            }
        }
        stage('promote image to dev') {
            //agent { docker { image "$DOCKER_AGENT" } }
            steps {
                script {
                    docker.withRegistry("https://${REGISTRY_DEV}",'registry-dev'){
                        shell("""docker push $BUILD_DEV """)
                        info('image promoted to dev')
                    }
                }
            }
        }
        /*
        stage('promote image to prod') {
            //agent { docker { image "$DOCKER_AGENT" } }
            steps {
                script {
                    docker.withRegistry("https://${REGISTRY_PROD}",'registry-prod'){
                        shell("""docker tag $BUILD_DEV $BUILD_PROD """)
                        shell("""docker push $BUILD_PROD """)
                        info('image promoted to prod')
                    }
                }
            }
        } */
        stage('exit') {
            steps {
                script {
                    info("logs :${BUILD_URL}console")
                    info("pipeline SUCCESS")
                    slack()
                }
            }
        }
    }
}

// Log into a file
def setLogger(){
    def RunLogger = LogManager.getLogManager().getLogger("global")
    def logsDir = new File(Jenkins.instance.rootDir, "logs")
    if(!logsDir.exists()){logsDir.mkdirs()}
    env.LOGFILE = logsDir.absolutePath+'/default.log'
    FileHandler handler = new FileHandler("${env.LOGFILE}", 1024 * 1024, 10, true);
    handler.setFormatter(new SimpleFormatter());
    RunLogger.addHandler(handler)
}

import groovy.json.JsonOutput;

class Event {
    def event
    def id
    def service
    def status
    def infos
}

def publish(String id, String status, String infos){
    Event evt = new Event('event':'ci', 'id':id, 'service':'bitbucket', 'status':status, 'infos':infos)
    def message = JsonOutput.toJson(evt)
    sh"""
        gcloud pubsub topics publish notifications-prod --message ${message}
    """
}

def header(){
    env.SLACK_HEADER = '[INFO] \n- push on branch <'+env.GIT_BRANCH+'>\n'+'- author <'+env.GIT_COMMITTER_NAME+'>\n'+'- email <'+env.GIT_COMMITTER_EMAIL+'>'
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

def readLogs(){
    try {
        def logs = readFile(env.LOGFILE)
        return logs
    }
    catch(e){
        def logs = "-- no logs --"
        return logs
    }
}

def shell(String command){
    try {
/*
        def output = sh(returnStatus: true, script: "${command} >${LOGFILE} 2>&1")
        if (output != 0){throw new Exception("Pipeline failed\n- command:: "+command)}
        else { return output }
*/
        sh("${command}")
    }
    catch (error){
        error(error.getMessage())
        error("- logs: ${BUILD_URL}console")
        error('Pipeline FAILED')
        slack()
        sh 'echo *****'
        sh 'echo ERROR'
        sh 'echo *****'
        //readLogs()
        throw error
    }
}