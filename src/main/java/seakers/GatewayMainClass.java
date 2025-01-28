package seakers;

import com.mathworks.engine.EngineException;
import py4j.GatewayServer;
import seakers.gatewayclasses.DesignOperations;
import seakers.gatewayclasses.eoss.EOSSDesignOperations;
import seakers.gatewayclasses.metamaterial.MetamaterialDesignOperations;

import java.util.concurrent.ExecutionException;

public class GatewayMainClass {

    private DesignOperations designOperations;

    public GatewayMainClass() throws ExecutionException, InterruptedException {
        String problemType = "eoss"; // Whether a metamaterial or EOSS optimization problem is to be solved

        switch (problemType) {
            case "metamaterial": {
                this.designOperations = new MetamaterialDesignOperations();
                break;
            }
            case "eoss": {
                this.designOperations = new EOSSDesignOperations();
            }
        }

    }

    public DesignOperations getOperationsInstance() {
        return this.designOperations;
    }

    public static void main(String[] args) throws ExecutionException, InterruptedException {
        GatewayServer gatewayServer = new GatewayServer(new GatewayMainClass());
        gatewayServer.start();
        System.out.println("Gateway Server started");
    }

}
