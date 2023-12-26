package seakers.gatewaytest;

import java.util.ArrayList;

public class TestIntegerProblemDesign {
    private int decisionTotal;
    private int[] currentDesign;

    public void setCurrentDesign(ArrayList<Integer> design) {
        this.currentDesign = design.stream().mapToInt(i->i).toArray();
    }

    public int[] getCurrentDesign() {
        return this.currentDesign;
    }

    public void computeTotal() {
        int total = 0;
        for (int decision : currentDesign) {
            total += decision;
        }
        this.decisionTotal = total;
    }

    public int getDecisionTotal() {
        return this.decisionTotal;
    }

}
