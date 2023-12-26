package seakers.gatewaytest;

import java.util.ArrayList;

public class TestBinaryProblemDesign {
    private int decisionTotal;
    private boolean[] currentDesign;

    public void setCurrentDesign(ArrayList<Boolean> design) {
        this.currentDesign = new boolean[design.size()];
        for (int i = 0; i < design.size(); i++) {
            this.currentDesign[i] = design.get(i);
        }
    }

    public boolean[] getCurrentDesign() {
        return this.currentDesign;
    }

    public void computeTotal() {
        int total = 0;
        for (boolean decision : currentDesign) {
            if (decision) {
                total += 1;
            }
        }
        this.decisionTotal = total;
    }

    public int getDecisionTotal() {
        return this.decisionTotal;
    }

}
