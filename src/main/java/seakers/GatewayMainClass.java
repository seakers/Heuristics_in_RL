package seakers;

import com.mathworks.engine.EngineException;
import py4j.GatewayServer;
import seakers.gatewayclasses.metamaterial.MetamaterialDesignOperations;

import java.util.concurrent.ExecutionException;

public class GatewayMainClass {

    private MetamaterialDesignOperations designOperations;

    public GatewayMainClass() throws ExecutionException, InterruptedException {
        this.designOperations = new MetamaterialDesignOperations();
    }

    public MetamaterialDesignOperations getOperationsInstance() {
        return this.designOperations;
    }

    public static void main(String[] args) throws ExecutionException, InterruptedException {
        GatewayServer gatewayServer = new GatewayServer(new GatewayMainClass());
        gatewayServer.start();
        System.out.println("Gateway Server started");
    }

}
