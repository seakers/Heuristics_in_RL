package seakers.gatewaytest;

import py4j.GatewayServer;

import java.util.ArrayList;

public class GatewayIntegerTestMainStart {

    private TestIntegerProblemDesign currentTestProblemDesign;

    public GatewayIntegerTestMainStart() {
        currentTestProblemDesign = new TestIntegerProblemDesign();
        ArrayList<Integer> startDesign = new ArrayList<>();
        currentTestProblemDesign.setCurrentDesign(startDesign);
    }

    public TestIntegerProblemDesign getCurrentTestProblemDesign() {
        return this.currentTestProblemDesign;
    }

    public static void main(String[] args) {
        GatewayServer gatewayServer = new GatewayServer(new GatewayIntegerTestMainStart());
        gatewayServer.start();
        System.out.println("Gateway Server started");
    }

}
