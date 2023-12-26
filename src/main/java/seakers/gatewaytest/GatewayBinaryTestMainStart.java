package seakers.gatewaytest;

import py4j.GatewayServer;

import java.util.ArrayList;

public class GatewayBinaryTestMainStart {

    private TestBinaryProblemDesign currentTestProblemDesign;

    public GatewayBinaryTestMainStart() {
        currentTestProblemDesign = new TestBinaryProblemDesign();
        ArrayList<Boolean> startDesign = new ArrayList<>();
        for (int i = 0; i < 5; i++) {
            startDesign.add(Boolean.FALSE);
        }
        currentTestProblemDesign.setCurrentDesign(startDesign);
    }

    public TestBinaryProblemDesign getCurrentTestProblemDesign() {
        return this.currentTestProblemDesign;
    }

    public static void main(String[] args) {
        GatewayServer gatewayServer = new GatewayServer(new GatewayBinaryTestMainStart());
        gatewayServer.start();
        System.out.println("Gateway Server started");
    }
}
