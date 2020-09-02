import java.util.logging.FileHandler
import java.util.logging.SimpleFormatter
import java.util.logging.LogManager
import jenkins.model.Jenkins

// hello 123

pipeline {
    agent { label 'azure-gpu' }
    agent { docker { image 'registryupstridedev.azurecr.io/ops:azure-cloud' } }
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
                    env.SLACK_HEADER = '[INFO] \n- push on branch <'+env.GIT_BRANCH+'>\n'+'- author <'+env.GIT_COMMITTER_NAME+'>\n'+'- email <'+env.GIT_COMMITTER_EMAIL+'>'
                    env.SLACK_MESSAGE = ''
                    env.BUILD_TAG = "upstride-python"
                    env.BUILD_VERSION = readFile("version")
                }
                setLogger()
                info("Starting the pipeline")
            }
        }
        stage('build docker image') {
            steps {
                script {
                    docker.withRegistry("https://${REGISTRY_DEV}",'registry-dev'){
                        shell("""docker build . -f dockerfile -t ${REGISTRY_DEV}/${REPO}:${BUILD_TAG}-${BUILD_VERSION} """)
                        info('built successful')
                    }
                }
            }
        }
        stage('promote image to dev') {
            steps {
                script {
                    docker.withRegistry("https://${REGISTRY_DEV}",'registry-dev'){
                        shell("""docker push ${REGISTRY_DEV}/upstride:${BUILD_TAG}-${BUILD_VERSION}""")
                        info('image promoted to dev')
                    }
                }
            }
        }
        stage('smoke tests') {
            options {
                timeout(time: 300, unit: "SECONDS")
            }
            agent { docker { image '${REGISTRY_DEV}/upstride:${BUILD_TAG}-${BUILD_VERSION}' } }
            steps {
                script {
                    shell("""pip install .""")
                    tests = ['test.py', 'test_tf.py', 'test_type1.py','test_type2.py', 'test_type3.py']
                    for (int i = 0; i < tests.size(); i++) {
                        shell("""python3 ${tests[i]}""")
                    }
                    info('tests cleared')
                }
            }
        }
        stage('promote image to prod') {
            steps {
                script {
                    docker.withRegistry("https://${REGISTRY_PROD}",'registry-prod'){
                        shell("""docker tag ${REGISTRY_DEV}/${REPO}:${BUILD_TAG}-${BUILD_VERSION} ${REGISTRY_PROD}/${REPO}:${BUILD_TAG}-${BUILD_VERSION} """)
                        shell("""docker push ${REGISTRY_PROD}/upstride:${BUILD_TAG}-${BUILD_VERSION}""")
                        info('image promoted to prod')
                    }
                }
            }
        }
        stage('exit') {
            steps {
                script {
                    info("logs :${BUILD_URL}console")
                    slack("[INFO] pipeline SUCCESS")
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

def slack(String body){
    env.SLACK_MESSAGE = env.SLACK_MESSAGE+'\n'+body.toString()
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
        error('Pipeline FAILED')
        slack("[ERROR] "+error.getMessage()+"\n- logs: ${BUILD_URL}console")
        //readLogs()
        throw error
    }
}