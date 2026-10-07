import ij.IJ;

import ij.ImagePlus;

import ij.WindowManager;

import ij.plugin.PlugIn;

import ij.process.ColorProcessor;



import java.awt.AlphaComposite;

import java.awt.BorderLayout;

import java.awt.Color;

import java.awt.Dimension;

import java.awt.Graphics2D;

import java.awt.GridLayout;

import java.awt.RenderingHints;

import java.awt.image.BufferedImage;

import java.util.ArrayList;

import java.util.List;

import java.util.Random;



import javax.swing.BorderFactory;

import javax.swing.ButtonGroup;

import javax.swing.JButton;

import javax.swing.JFrame;

import javax.swing.JLabel;

import javax.swing.JPanel;

import javax.swing.JSlider;

import javax.swing.JToggleButton;

import javax.swing.SwingUtilities;

import javax.swing.SwingWorker;

import javax.swing.event.ChangeEvent;

import javax.swing.event.ChangeListener;



import com.jogamp.opengl.GL;

import com.jogamp.opengl.GL2ES2;

import com.jogamp.opengl.GL3;

import com.jogamp.opengl.GLAutoDrawable;

import com.jogamp.opengl.GLCapabilities;

import com.jogamp.opengl.GLEventListener;

import com.jogamp.opengl.GLProfile;

import com.jogamp.opengl.awt.GLCanvas;

import com.jogamp.opengl.util.FPSAnimator;



import org.apache.commons.math3.linear.EigenDecomposition;

import org.apache.commons.math3.linear.LUDecomposition;

import org.apache.commons.math3.linear.MatrixUtils;

import org.apache.commons.math3.linear.RealMatrix;

import org.apache.commons.math3.linear.SingularValueDecomposition;



public class Rascal_GL implements PlugIn {



    static final String VERSION = "1.6.1";



    @Override

    public void run(String arg) {

        ImagePlus imp = WindowManager.getCurrentImage();

        if (imp == null) {

            IJ.noImage();

            return;

        }



        if (imp.getType() != ImagePlus.COLOR_RGB) {

            IJ.error("Please open an RGB image first.");

            return;

        }



        ColorProcessor cp = (ColorProcessor) imp.getProcessor().convertToColorProcessor();

        BufferedImage buffered = cp.getBufferedImage();

        int width = cp.getWidth();

        int height = cp.getHeight();



        float[] linearRGB = extractLinearRGB(cp);

        double[] mu = computeMean(linearRGB);

        double[][] cov = computeCovariance(linearRGB, mu);

        ZCAResult zca = computeZCA(cov, 1e-6);



        SwingUtilities.invokeLater(() -> {

            try {

                GLProfile.initSingleton();

            } catch (Throwable t) {

                IJ.error("RASCAL GL",

                    "GLProfile.initSingleton() failed.\n" +

                    "OpenGL/JOGL cannot be initialized on this system.\n\n" +

                    t.toString());

                return;

            }

            PreviewFrame frame = new PreviewFrame(

                    imp.getTitle(),

                    buffered,

                    linearRGB,

                    width,

                    height,

                    mu,

                    zca.W,

                    zca.Winv

            );

            frame.setVisible(true);

        });

    }



    // ============================================================

    // UI

    // ============================================================



    static class PreviewFrame extends JFrame {

        private final GLPreviewCanvas canvas;

        private final FPSAnimator animator;

        private final float[] linearRGB;

        private final int width;

        private final int height;

        private final String sourceTitle;

        private final double[] origMu;

        private final double[][] origW;

        private final double[][] origWinv;

        private double[] icaMu;

        private double[][] icaW;

        private double[][] icaWinv;



        private BufferedImage brushImage = null;

        private boolean paintMode = false;

        private JToggleButton roiBtn;

        private JToggleButton icaBtn;

        private JButton polBtn;

        private boolean suppressIcaReset  = false;

        private volatile boolean suppressRandomStop = false;

        private volatile boolean randomRunning = false;

        private javax.swing.Timer randomTimer = null;

        private JToggleButton randomBtn;

        private final Random rng = new Random();

        private JButton clearPresetsBtn;



        private final JLabel labelX    = new JLabel("X: 0°");

        private final JLabel labelY    = new JLabel("Y: 0°");

        private final JLabel labelZ    = new JLabel("Z: 0°");

        private final JLabel labelPush = new JLabel("Push: 1.00");

        private final JLabel labelVar  = new JLabel("Var: 1.00");



        private final JSlider sliderX    = new JSlider(-180, 180, 0);

        private final JSlider sliderY    = new JSlider(-180, 180, 0);

        private final JSlider sliderZ    = new JSlider(-180, 180, 0);

        private final JSlider sliderPush = new JSlider(0, 500, 100);

        private final JSlider sliderVar  = new JSlider(0, 100, 100);



        private static class Preset {

            final int[] sliders;

            final double[] mu;

            final double[][] W, Winv;

            final boolean roiActive;

            Preset(int[] s, double[] mu, double[][] W, double[][] Winv, boolean roi) {

                this.sliders = s; this.mu = mu; this.W = W; this.Winv = Winv;

                this.roiActive = roi;

            }

        }

        private static double[][] copyMat(double[][] m) {

            double[][] c = new double[m.length][];

            for (int i = 0; i < m.length; i++) c[i] = m[i].clone();

            return c;

        }



        private final List<Preset> presets = new ArrayList<>();

        private JPanel presetPanel;



        private int lastMouseX = 0;

        private int lastMouseY = 0;

        private static final float SENS_DRAG  = 0.4f;

        private static final float SENS_WHEEL = 3.0f;

        private volatile boolean ctrlDown = false;

        private java.awt.KeyEventDispatcher keyDispatcher;



        PreviewFrame(String title,

                     BufferedImage buffered,

                     float[] linearRGB,

                     int width,

                     int height,

                     double[] mu,

                     double[][] W,

                     double[][] Winv) {

            super("RASCAL GL v" + VERSION + " - " + title);



            this.sourceTitle = title;

            this.linearRGB = linearRGB;

            this.width = width;

            this.height = height;

            this.origMu = mu; this.origW = W; this.origWinv = Winv;

            this.icaMu = mu; this.icaW = W; this.icaWinv = Winv;



            GLPreviewCanvas tmp;

            try {

                tmp = new GLPreviewCanvas(buffered, mu, W, Winv);

            } catch (Throwable t) {

                IJ.error("RASCAL GL",

                    "OpenGL/JOGL could not be initialized.\n\n" +

                    "This is probably a JOGL native-library problem on this Mac.\n" +

                    "Try using the Fiji version, or update JOGL/GlueGen to 2.5.0 or later.\n\n" +

                    t.toString());

                throw new RuntimeException("JOGL init failed", t);

            }

            this.canvas = tmp;

            this.canvas.setPreferredSize(new Dimension(900, 700));



            setLayout(new BorderLayout());

            add(canvas, BorderLayout.CENTER);

            add(buildControls(), BorderLayout.SOUTH);

            saveCurrentAsPreset();



            this.animator = new FPSAnimator(canvas, 60);

            this.canvas.setOnInit(() -> { if (!animator.isAnimating()) animator.start(); });



            setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);

            pack();

            setLocationRelativeTo(null);

            addWindowListener(new java.awt.event.WindowAdapter() {

                @Override

                public void windowClosing(java.awt.event.WindowEvent e) {

                    animator.stop();

                    java.awt.KeyboardFocusManager.getCurrentKeyboardFocusManager()

                            .removeKeyEventDispatcher(keyDispatcher);

                }

            });



            ChangeListener listener = new ChangeListener() {

                @Override

                public void stateChanged(ChangeEvent e) {

                    int ax = sliderX.getValue();

                    int ay = sliderY.getValue();

                    int az = sliderZ.getValue();

                    labelX.setText("X: " + ax + "°");

                    labelY.setText("Y: " + ay + "°");

                    labelZ.setText("Z: " + az + "°");

                    canvas.setAnglesDegrees(ax, ay, az);

                    if (!suppressIcaReset && icaBtn != null) icaBtn.setSelected(false);

                    if (!suppressRandomStop && randomRunning) stopRandom();

                }

            };





            sliderX.addChangeListener(listener);

            sliderY.addChangeListener(listener);

            sliderZ.addChangeListener(listener);



            canvas.addMouseListener(new java.awt.event.MouseAdapter() {

                @Override

                public void mousePressed(java.awt.event.MouseEvent e) {

                    lastMouseX = e.getX();

                    lastMouseY = e.getY();

                    

                    // Check if clicking near split line

                    if (canvas.isSplitEnabled() && SwingUtilities.isLeftMouseButton(e)) {

                        float splitScreenX = getSplitScreenX();

                        if (Math.abs(e.getX() - splitScreenX) < 15) {

                            canvas.setSplitDragging(true);

                            return;

                        }

                    }

                    

                    if (paintMode && e.getClickCount() == 2

                            && SwingUtilities.isLeftMouseButton(e)) {

                        validateROI();

                    }

                }

                

                @Override

                public void mouseReleased(java.awt.event.MouseEvent e) {

                    canvas.setSplitDragging(false);

                }

            });



            canvas.addMouseMotionListener(new java.awt.event.MouseMotionAdapter() {

                @Override

                public void mouseDragged(java.awt.event.MouseEvent e) {

                    // Handle split line dragging

                    if (canvas.isSplitDragging()) {

                        float[] scale = canvas.getAspectScale();

                        int cw = canvas.getWidth();

                        float sx = scale[0];

                        float zs = canvas.getZoomScale();

                        float zox = canvas.getZoomOffX();

                        

                        // Calculate image bounds on screen

                        int imgScreenW = (int) (sx * zs * cw);

                        float ndcLeft = -sx * zs + zox;

                        float screenLeft = (ndcLeft + 1f) * cw / 2f;

                        

                        // Convert mouse X to texture coordinate (0-1)

                        float texX = (e.getX() - screenLeft) / imgScreenW;

                        canvas.setSplitPosition(texX);

                        

                        // Update cursor

                        canvas.setCursor(java.awt.Cursor.getPredefinedCursor(java.awt.Cursor.E_RESIZE_CURSOR));

                        return;

                    }

                    

                    int dx = e.getX() - lastMouseX;

                    int dy = e.getY() - lastMouseY;

                    lastMouseX = e.getX();

                    lastMouseY = e.getY();

                    if (paintMode) {

                        if (javax.swing.SwingUtilities.isLeftMouseButton(e)) {

                            paintBrushAt(e.getX(), e.getY());

                            lastMouseX = e.getX(); lastMouseY = e.getY();

                        }

                        return;

                    }

                    if (javax.swing.SwingUtilities.isLeftMouseButton(e)) {

                        sliderZ.setValue(wrapAngle(sliderZ.getValue() + Math.round(dx * SENS_DRAG)));

                    } else if (javax.swing.SwingUtilities.isRightMouseButton(e)) {

                        sliderY.setValue(wrapAngle(sliderY.getValue() + Math.round(dx * SENS_DRAG)));

                    } else if (javax.swing.SwingUtilities.isMiddleMouseButton(e)) {

                        sliderX.setValue(wrapAngle(sliderX.getValue() - Math.round(dy * SENS_DRAG)));

                    }

                }

                

                @Override

                public void mouseMoved(java.awt.event.MouseEvent e) {

                    // Update cursor when hovering near split line

                    if (canvas.isSplitEnabled()) {

                        float splitScreenX = getSplitScreenX();

                        if (Math.abs(e.getX() - splitScreenX) < 15) {

                            canvas.setCursor(java.awt.Cursor.getPredefinedCursor(java.awt.Cursor.E_RESIZE_CURSOR));

                            return;

                        }

                    }

                    canvas.setCursor(java.awt.Cursor.getDefaultCursor());

                }

            });



            canvas.addMouseWheelListener(e -> {

                if (ctrlDown) {

                    canvas.applyZoom(e.getPreciseWheelRotation(), e.getX(), e.getY());

                    return;

                }

                if (paintMode) return;

                int delta = (int) Math.round(e.getPreciseWheelRotation() * SENS_WHEEL);

                sliderX.setValue(wrapAngle(sliderX.getValue() + delta));

            });



            keyDispatcher = ke -> {

                if (ke.getKeyCode() == java.awt.event.KeyEvent.VK_CONTROL) {

                    ctrlDown = (ke.getID() == java.awt.event.KeyEvent.KEY_PRESSED);

                }

                return false;

            };

            java.awt.KeyboardFocusManager.getCurrentKeyboardFocusManager()

                    .addKeyEventDispatcher(keyDispatcher);

        }



        private static int wrapAngle(int v) {

            while (v > 180)  v -= 360;

            while (v < -180) v += 360;

            return v;

        }

        

        // Calculate screen X coordinate of the split line

        private float getSplitScreenX() {

            if (!canvas.isSplitEnabled()) return -1;

            

            float[] scale = canvas.getAspectScale();

            int cw = canvas.getWidth();

            float sx = scale[0];

            float zs = canvas.getZoomScale();

            float zox = canvas.getZoomOffX();

            

            // Image width in NDC space (-1 to 1)

            float imgNDCWidth = 2f * sx * zs;

            float imgLeftNDC = -imgNDCWidth / 2f + zox;

            

            // Convert split position (0-1) to NDC X

            float splitNDC = imgLeftNDC + canvas.getSplitPosition() * imgNDCWidth;

            

            // Convert NDC to screen coordinates

            return (splitNDC + 1f) * cw / 2f;

        }



        private JPanel buildControls() {

            JPanel root = new JPanel(new BorderLayout());

            root.setBorder(BorderFactory.createEmptyBorder(8, 8, 8, 8));



            sliderPush.addChangeListener(e -> {

                float p = sliderPush.getValue() / 100f;

                labelPush.setText(String.format("Push: %.2f", p));

                canvas.setPush(p);

                if (icaBtn != null) icaBtn.setSelected(false);

                if (randomRunning) stopRandom();

            });



            sliderVar.addChangeListener(e -> {

                float v = sliderVar.getValue() / 100f;

                labelVar.setText(String.format("Var: %.2f", v));

                canvas.setVarRestore(v);

                if (icaBtn != null) icaBtn.setSelected(false);

                if (randomRunning) stopRandom();

            });



            JPanel sliders = new JPanel(new GridLayout(5, 2, 8, 6));

            sliders.add(labelVar);

            sliders.add(sliderVar);

            sliders.add(labelPush);

            sliders.add(sliderPush);

            sliders.add(labelX);

            sliders.add(sliderX);

            sliders.add(labelY);

            sliders.add(sliderY);

            sliders.add(labelZ);

            sliders.add(sliderZ);



            JButton reset = new JButton("Reset");

            reset.setToolTipText("Reset all sliders and zoom (shortcut: R)");

            reset.addActionListener(e -> {

                sliderX.setValue(0);

                sliderY.setValue(0);

                sliderZ.setValue(0);

                sliderPush.setValue(100);

                sliderVar.setValue(100);

                canvas.resetZoom();

            });



            icaBtn = new JToggleButton("ICA");

            icaBtn.setToolTipText("Find independent colour axes via FastICA (shortcut: I)");

            icaBtn.addActionListener(e -> runICA(icaBtn));



            // Polarity inversion button

            polBtn = new JButton("Pol. +/-");

            polBtn.setToolTipText("Invert rotation polarity: negate X, Y, Z angles");

            polBtn.addActionListener(e -> invertPolarity());



            JButton apply = new JButton("Apply and save");

            apply.addActionListener(e -> applyCurrentTransform());



            roiBtn = new JToggleButton("ROI");

            roiBtn.setToolTipText("Left drag=paint, double-click=validate, click again=cancel (shortcut: O)");

            roiBtn.addActionListener(e -> {

                if (roiBtn.isSelected()) {

                    if (brushImage == null) {

                        brushImage = new BufferedImage(width, height, BufferedImage.TYPE_INT_ARGB);

                        canvas.setBrushImage(brushImage);

                    }

                    paintMode = true;

                    canvas.setPaintModeGL(true);

                } else {

                    cancelROI();

                }

            });



            JToggleButton btnC = new JToggleButton("C", true);

            JToggleButton btnR = new JToggleButton("R");

            JToggleButton btnG = new JToggleButton("G");

            JToggleButton btnB = new JToggleButton("B");

            ButtonGroup channelGroup = new ButtonGroup();

            channelGroup.add(btnC); channelGroup.add(btnR);

            channelGroup.add(btnG); channelGroup.add(btnB);

            btnC.addActionListener(e -> canvas.setViewMode(0));

            btnR.addActionListener(e -> canvas.setViewMode(1));

            btnG.addActionListener(e -> canvas.setViewMode(2));

            btnB.addActionListener(e -> canvas.setViewMode(3));



            JPanel buttons = new JPanel();

            buttons.add(btnC); buttons.add(btnR);

            buttons.add(btnG); buttons.add(btnB);

            buttons.add(roiBtn);

            buttons.add(icaBtn);

            buttons.add(polBtn);

            buttons.add(reset);

            randomBtn = new JToggleButton("Random");

            randomBtn.setToolTipText("Random rotations every 3 s — any manual action stops it (shortcut: Ctrl+Shift+R)");

            randomBtn.addActionListener(e -> {

                if (randomBtn.isSelected()) startRandom();

                else stopRandom();

            });

            buttons.add(randomBtn);

            

            // Split view button

            JToggleButton splitBtn = new JToggleButton("Split");

            splitBtn.setToolTipText("Split view: left=original, right=transformed (drag white line) (shortcut: T)");

            splitBtn.addActionListener(e -> canvas.setSplitEnabled(splitBtn.isSelected()));

            buttons.add(splitBtn);

            

            buttons.add(apply);



            presetPanel = new JPanel(new java.awt.FlowLayout(java.awt.FlowLayout.LEFT, 4, 2));

            JButton saveBtn = makePresetSaveBtn();

            presetPanel.add(saveBtn);

            clearPresetsBtn = new JButton("\u00d7");

            clearPresetsBtn.setToolTipText("Clear all presets except the first (original)");

            clearPresetsBtn.addActionListener(e -> clearPresets());



            JPanel top = new JPanel(new BorderLayout());

            top.add(presetPanel, BorderLayout.NORTH);

            top.add(buttons, BorderLayout.SOUTH);



            JLabel info = new JLabel("v" + VERSION + "  –  Fabrice.Monna@u-bourgogne.fr");

            info.setFont(info.getFont().deriveFont(java.awt.Font.ITALIC, 10f));

            info.setForeground(java.awt.Color.GRAY);



            JPanel infoPanel = new JPanel(new BorderLayout());

            infoPanel.setBorder(BorderFactory.createEmptyBorder(18, 0, 0, 0));

            infoPanel.add(info, BorderLayout.CENTER);



            root.add(top,      BorderLayout.NORTH);

            root.add(sliders,  BorderLayout.CENTER);

            root.add(infoPanel, BorderLayout.SOUTH);



            registerShortcuts(root, reset, splitBtn, saveBtn);



            return root;

        }



        // Window-wide keyboard shortcuts.

        private void registerShortcuts(JPanel root, JButton reset,

                                       JToggleButton splitBtn, JButton saveBtn) {

            javax.swing.InputMap im = root.getInputMap(JPanel.WHEN_IN_FOCUSED_WINDOW);

            javax.swing.ActionMap am = root.getActionMap();



            bindKey(im, am, javax.swing.KeyStroke.getKeyStroke(

                    java.awt.event.KeyEvent.VK_R, 0), "sc_reset",

                    e -> reset.doClick());

            bindKey(im, am, javax.swing.KeyStroke.getKeyStroke(

                    java.awt.event.KeyEvent.VK_T, 0), "sc_split",

                    e -> splitBtn.doClick());

            bindKey(im, am, javax.swing.KeyStroke.getKeyStroke(

                    java.awt.event.KeyEvent.VK_I, 0), "sc_ica",

                    e -> { if (icaBtn != null) icaBtn.doClick(); });

            bindKey(im, am, javax.swing.KeyStroke.getKeyStroke(

                    java.awt.event.KeyEvent.VK_P, 0), "sc_save",

                    e -> saveBtn.doClick());

            bindKey(im, am, javax.swing.KeyStroke.getKeyStroke(

                    java.awt.event.KeyEvent.VK_O, 0), "sc_roi",

                    e -> { if (roiBtn != null) roiBtn.doClick(); });

            bindKey(im, am, javax.swing.KeyStroke.getKeyStroke(

                    java.awt.event.KeyEvent.VK_R,

                    java.awt.event.InputEvent.CTRL_DOWN_MASK

                            | java.awt.event.InputEvent.SHIFT_DOWN_MASK),

                    "sc_random", e -> { if (randomBtn != null) randomBtn.doClick(); });

            bindKey(im, am, javax.swing.KeyStroke.getKeyStroke(

                    java.awt.event.KeyEvent.VK_UP, 0), "sc_push_up",

                    e -> sliderPush.setValue(

                            Math.min(sliderPush.getMaximum(), sliderPush.getValue() + 5)));

            bindKey(im, am, javax.swing.KeyStroke.getKeyStroke(

                    java.awt.event.KeyEvent.VK_DOWN, 0), "sc_push_down",

                    e -> sliderPush.setValue(

                            Math.max(sliderPush.getMinimum(), sliderPush.getValue() - 5)));

        }



        private void bindKey(javax.swing.InputMap im, javax.swing.ActionMap am,

                             javax.swing.KeyStroke ks, String name,

                             java.awt.event.ActionListener action) {

            im.put(ks, name);

            am.put(name, new javax.swing.AbstractAction() {

                @Override

                public void actionPerformed(java.awt.event.ActionEvent e) {

                    action.actionPerformed(e);

                }

            });

        }



        private void paintBrushAt(int mx, int my) {

            if (brushImage == null) return;

            float[] sc = canvas.getAspectScale();

            float sx = sc[0], sy = sc[1];

            int cw = canvas.getWidth(), ch = canvas.getHeight();

            if (cw <= 0 || ch <= 0) return;

            float zs = canvas.getZoomScale();

            float zox = canvas.getZoomOffX(), zoy = canvas.getZoomOffY();

            float ndcX = 2f * mx / cw - 1f;

            float ndcY = 1f - 2f * my / ch;

            float u = (ndcX - zox + sx * zs) / (2f * sx * zs);

            float v = (sy * zs + zoy - ndcY) / (2f * sy * zs);

            if (u < 0 || u > 1 || v < 0 || v > 1) return;

            int px = (int)(u * width);

            int py = (int)(v * height);

            int r = Math.max(8, (int)(15f * width / (sx * cw)));

            Graphics2D g2 = brushImage.createGraphics();

            g2.setComposite(AlphaComposite.getInstance(AlphaComposite.SRC_OVER, 1f));

            g2.setColor(new Color(255, 255, 0, 160));

            g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

            g2.fillOval(px - r, py - r, 2*r, 2*r);

            g2.dispose();

            canvas.markBrushDirty();

        }



        private void validateROI() {

            if (brushImage == null) return;

            List<Integer> roiIdx = new ArrayList<>();

            for (int py = 0; py < height; py++) {

                for (int px = 0; px < width; px++) {

                    int alpha = (brushImage.getRGB(px, py) >> 24) & 0xFF;

                    if (alpha > 64) roiIdx.add(py * width + px);

                }

            }

            if (roiIdx.size() < 20) {

                IJ.showStatus("ROI too small - paint a larger area");

                return;

            }

            float[] roiRGB = new float[roiIdx.size() * 3];

            for (int i = 0; i < roiIdx.size(); i++) {

                int k = roiIdx.get(i);

                roiRGB[3*i]   = linearRGB[3*k];

                roiRGB[3*i+1] = linearRGB[3*k+1];

                roiRGB[3*i+2] = linearRGB[3*k+2];

            }

            IJ.showStatus("Recomputing ZCA on " + roiIdx.size() + " pixels…");

            double[] newMu  = computeMean(roiRGB);

            double[][] newCov = computeCovariance(roiRGB, newMu);

            ZCAResult  zca  = computeZCA(newCov, 1e-6);

            icaMu = newMu; icaW = zca.W; icaWinv = zca.Winv;

            canvas.updateZCA(newMu, zca.W, zca.Winv);

            Graphics2D g2c = brushImage.createGraphics();

            g2c.setComposite(AlphaComposite.Clear);

            g2c.fillRect(0, 0, width, height);

            g2c.dispose();

            canvas.markBrushDirty();

            paintMode = false;

            canvas.setPaintModeGL(false);

            IJ.showStatus("ROI applied on " + roiIdx.size() + " pixels");

        }



        private void cancelROI() {

            if (brushImage != null) {

                Graphics2D g2 = brushImage.createGraphics();

                g2.setComposite(AlphaComposite.Clear);

                g2.fillRect(0, 0, width, height);

                g2.dispose();

                canvas.markBrushDirty();

            }

            icaMu = origMu; icaW = origW; icaWinv = origWinv;

            canvas.updateZCA(origMu, origW, origWinv);

            canvas.setPaintModeGL(false);

            paintMode = false;

            roiBtn.setSelected(false);

            IJ.showStatus("ROI cancelled");

        }



        private void startRandom() {

            randomRunning = true;

            if (icaBtn != null) icaBtn.setSelected(false);

            applyRandomRotation();

            randomTimer = new javax.swing.Timer(3000, e -> {

                if (randomRunning) applyRandomRotation();

            });

            randomTimer.setRepeats(true);

            randomTimer.start();

        }



        private void stopRandom() {

            randomRunning = false;

            if (randomTimer != null) { randomTimer.stop(); randomTimer = null; }

            if (randomBtn != null) randomBtn.setSelected(false);

        }



        private void applyRandomRotation() {

            suppressRandomStop = true;

            suppressIcaReset   = true;

            sliderX.setValue(rng.nextInt(361) - 180);

            sliderY.setValue(rng.nextInt(361) - 180);

            sliderZ.setValue(rng.nextInt(361) - 180);

            suppressIcaReset   = false;

            suppressRandomStop = false;

        }



        private void invertPolarity() {

            suppressIcaReset = true;

            int newX = wrapAngle(-sliderX.getValue());

            int newY = wrapAngle(-sliderY.getValue());

            int newZ = wrapAngle(-sliderZ.getValue());

            sliderX.setValue(newX);

            sliderY.setValue(newY);

            sliderZ.setValue(newZ);

            suppressIcaReset = false;

            IJ.showStatus("Polarity inverted: X=" + newX + "° Y=" + newY + "° Z=" + newZ + "°");

        }



        private void saveCurrentAsPreset() {

            if (presets.size() >= 10) {

                javax.swing.JOptionPane.showMessageDialog(this,

                    "Maximum 10 presets reached.", "Presets",

                    javax.swing.JOptionPane.INFORMATION_MESSAGE);

                return;

            }

            Preset p = new Preset(

                new int[]{ sliderX.getValue(), sliderY.getValue(), sliderZ.getValue(),

                           sliderPush.getValue(), sliderVar.getValue() },

                icaMu.clone(), copyMat(icaW), copyMat(icaWinv),

                roiBtn != null && roiBtn.isSelected()

            );

            presets.add(p);

            int idx = presets.size();

            JButton btn = new JButton(String.valueOf(idx));

            btn.setToolTipText(String.format(

                "X:%d\u00b0 Y:%d\u00b0 Z:%d\u00b0 Push:%.2f Var:%.2f%s",

                p.sliders[0], p.sliders[1], p.sliders[2],

                p.sliders[3]/100f, p.sliders[4]/100f,

                p.roiActive ? " [ROI]" : ""));

            btn.addActionListener(ev -> restorePreset(idx - 1));

            if (clearPresetsBtn != null) presetPanel.remove(clearPresetsBtn);

            presetPanel.add(btn);

            if (clearPresetsBtn != null) presetPanel.add(clearPresetsBtn);

            presetPanel.revalidate();

            presetPanel.repaint();

        }



        private void clearPresets() {

            if (presets.size() <= 1) return;

            while (presets.size() > 1) presets.remove(presets.size() - 1);

            while (presetPanel.getComponentCount() > 1) presetPanel.remove(1);

            Preset p0 = presets.get(0);

            JButton btn = new JButton("1");

            btn.setToolTipText(String.format(

                "X:%d\u00b0 Y:%d\u00b0 Z:%d\u00b0 Push:%.2f Var:%.2f%s",

                p0.sliders[0], p0.sliders[1], p0.sliders[2],

                p0.sliders[3]/100f, p0.sliders[4]/100f,

                p0.roiActive ? " [ROI]" : ""));

            btn.addActionListener(ev -> restorePreset(0));

            presetPanel.add(btn);

            presetPanel.add(clearPresetsBtn);

            presetPanel.revalidate();

            presetPanel.repaint();

        }



        private JButton makePresetSaveBtn() {

            JButton save = new JButton("Save preset");

            save.addActionListener(e -> saveCurrentAsPreset());

            return save;

        }



        private void restorePreset(int i) {

            if (i < 0 || i >= presets.size()) return;

            Preset p = presets.get(i);

            sliderX.setValue(p.sliders[0]);

            sliderY.setValue(p.sliders[1]);

            sliderZ.setValue(p.sliders[2]);

            sliderPush.setValue(p.sliders[3]);

            sliderVar.setValue(p.sliders[4]);

            icaMu = p.mu.clone();

            icaW  = copyMat(p.W);

            icaWinv = copyMat(p.Winv);

            canvas.updateZCA(icaMu, icaW, icaWinv);

            roiBtn.setSelected(p.roiActive);

            paintMode = false;

            canvas.setPaintModeGL(false);

        }



        private void applyCurrentTransform() {

            float[] RW   = canvas.getCurrentRW();

            float   push = canvas.getPush();

            float   var  = canvas.getVarRestore();



            float[] Wf = flatten3x3f(icaWinv);

            float[] effWinv = new float[9];

            for (int k = 0; k < 9; k++) {

                float id = (k % 4 == 0) ? 1f : 0f;

                effWinv[k] = id + var * (Wf[k] - id);

            }



            float mu0 = (float) icaMu[0], mu1 = (float) icaMu[1], mu2 = (float) icaMu[2];



            int[] out = new int[width * height];



            for (int i = 0; i < width * height; i++) {

                float r = linearRGB[3 * i];

                float g = linearRGB[3 * i + 1];

                float bl = linearRGB[3 * i + 2];



                float xr = r - mu0, xg = g - mu1, xb = bl - mu2;

                float zr = push * (RW[0]*xr + RW[1]*xg + RW[2]*xb);

                float zg = push * (RW[3]*xr + RW[4]*xg + RW[5]*xb);

                float zb = push * (RW[6]*xr + RW[7]*xg + RW[8]*xb);



                float rr = mu0 + effWinv[0]*zr + effWinv[1]*zg + effWinv[2]*zb;

                float gg = mu1 + effWinv[3]*zr + effWinv[4]*zg + effWinv[5]*zb;

                float bb = mu2 + effWinv[6]*zr + effWinv[7]*zg + effWinv[8]*zb;



                rr = clamp01(rr);

                gg = clamp01(gg);

                bb = clamp01(bb);



                int rs = linearToSrgb8(rr);

                int gs = linearToSrgb8(gg);

                int bs = linearToSrgb8(bb);



                out[i] = (rs << 16) | (gs << 8) | bs;

            }



            String ts = new java.text.SimpleDateFormat("yyyyMMdd_HHmm")

                    .format(new java.util.Date());

            String saveName = "Rascal_" + sourceTitle + "_" + ts;



            ColorProcessor cpOut = new ColorProcessor(width, height, out);

            ImagePlus outImp = new ImagePlus(saveName, cpOut);

            outImp.show();



            ImagePlus srcImp = WindowManager.getImage(sourceTitle);

            String dir = (srcImp != null && srcImp.getOriginalFileInfo() != null)

                    ? srcImp.getOriginalFileInfo().directory : null;

            String savePath = (dir != null) ? dir + saveName : saveName;

            IJ.saveAs(outImp, "Jpeg", savePath);

            IJ.showStatus("Saved: " + saveName + ".jpg");

        }



        private void runICA(JToggleButton btn) {

            btn.setEnabled(false);



            new SwingWorker<int[], Void>() {

                @Override

                protected int[] doInBackground() {

                    int n = linearRGB.length / 3;

                    int nSamples = Math.min(100_000, n);



                    // Random sample (Fisher-Yates partial shuffle, seed=0)

                    int[] indices = new int[n];

                    for (int i = 0; i < n; i++) indices[i] = i;

                    Random rng = new Random(0);

                    for (int i = 0; i < nSamples; i++) {

                        int j = i + rng.nextInt(n - i);

                        int tmp = indices[i]; indices[i] = indices[j]; indices[j] = tmp;

                    }



                    // Z = (X - mu) @ W.T   (whitened data)

                    double[][] Z = new double[nSamples][3];

                    for (int s = 0; s < nSamples; s++) {

                        int idx = indices[s];

                        double r = linearRGB[3 * idx]     - icaMu[0];

                        double g = linearRGB[3 * idx + 1] - icaMu[1];

                        double b = linearRGB[3 * idx + 2] - icaMu[2];

                        for (int j = 0; j < 3; j++) {

                            Z[s][j] = icaW[j][0]*r + icaW[j][1]*g + icaW[j][2]*b;

                        }

                    }



                    // FastICA deflation → rotation matrix

                    double[][] W_ica = fastICADeflation(Z);



                    // SVD to enforce orthogonality

                    RealMatrix Rm = MatrixUtils.createRealMatrix(W_ica);

                    SingularValueDecomposition svd = new SingularValueDecomposition(Rm);

                    RealMatrix R = svd.getU().multiply(svd.getVT());



                    // Ensure det(R) = +1

                    if (new LUDecomposition(R).getDeterminant() < 0) {

                        double[] row = R.getRow(2);

                        for (int j = 0; j < 3; j++) row[j] = -row[j];

                        R = R.copy();

                        R.setRow(2, row);

                    }



                    // Euler ZYX decomposition → degrees

                    double[] angles = rToEulerZYX(R.getData());

                    int ax = (int) Math.round(Math.toDegrees(angles[0]));

                    int ay = (int) Math.round(Math.toDegrees(angles[1]));

                    int az = (int) Math.round(Math.toDegrees(angles[2]));

                    return new int[]{wrapAngle(ax), wrapAngle(ay), wrapAngle(az)};

                }



                @Override

                protected void done() {

                    btn.setEnabled(true);

                    try {

                        int[] angles = get();

                        suppressIcaReset = true;

                        sliderX.setValue(angles[0]);

                        sliderY.setValue(angles[1]);

                        sliderZ.setValue(angles[2]);

                        suppressIcaReset = false;

                        icaBtn.setSelected(true);

                    } catch (Exception ex) {

                        IJ.error("ICA failed: " + ex.getMessage());

                    }

                }

            }.execute();

        }



        private static double[][] fastICADeflation(double[][] Z) {

            int n = Z.length;

            int p = 3;

            double[][] W_ica = new double[p][p];



            for (int i = 0; i < p; i++) {

                double[] w = new double[p];

                Random rng = new Random(i);

                for (int k = 0; k < p; k++) w[k] = rng.nextGaussian();

                vecNormalize(w);



                for (int iter = 0; iter < 200; iter++) {

                    // g = tanh(Z @ w),  gpMean = mean(1 - tanh²)

                    double[] g   = new double[n];

                    double gpMean = 0;

                    for (int s = 0; s < n; s++) {

                        double proj = 0;

                        for (int k = 0; k < p; k++) proj += Z[s][k] * w[k];

                        double t = Math.tanh(proj);

                        g[s] = t;

                        gpMean += 1.0 - t * t;

                    }

                    gpMean /= n;



                    // w_new = (Z.T @ g) / n  -  gpMean * w

                    double[] wNew = new double[p];

                    for (int k = 0; k < p; k++) {

                        double acc = 0;

                        for (int s = 0; s < n; s++) acc += Z[s][k] * g[s];

                        wNew[k] = acc / n - gpMean * w[k];

                    }



                    // Gram-Schmidt deflation against previous components

                    for (int j = 0; j < i; j++) {

                        double dot = 0;

                        for (int k = 0; k < p; k++) dot += wNew[k] * W_ica[j][k];

                        for (int k = 0; k < p; k++) wNew[k] -= dot * W_ica[j][k];

                    }



                    double norm = vecNorm(wNew);

                    if (norm < 1e-12) break;

                    for (int k = 0; k < p; k++) wNew[k] /= norm;



                    double dot = 0;

                    for (int k = 0; k < p; k++) dot += wNew[k] * w[k];

                    boolean converged = Math.abs(Math.abs(dot) - 1.0) < 1e-6;

                    w = wNew;

                    if (converged) break;

                }

                W_ica[i] = w;

            }

            return W_ica;

        }



        private static double[] rToEulerZYX(double[][] R) {

            double sy = Math.sqrt(R[0][0]*R[0][0] + R[1][0]*R[1][0]);

            double ax, ay, az;

            if (sy > 1e-6) {

                ax = Math.atan2( R[2][1], R[2][2]);

                ay = Math.atan2(-R[2][0], sy);

                az = Math.atan2( R[1][0], R[0][0]);

            } else {

                ax = Math.atan2(-R[1][2], R[1][1]);

                ay = Math.atan2(-R[2][0], sy);

                az = 0.0;

            }

            return new double[]{ax, ay, az};

        }



        private static void vecNormalize(double[] v) {

            double n = vecNorm(v);

            if (n > 1e-12) for (int i = 0; i < v.length; i++) v[i] /= n;

        }



        private static double vecNorm(double[] v) {

            double s = 0; for (double x : v) s += x*x; return Math.sqrt(s);

        }

    }



    // ============================================================

    // OpenGL preview canvas

    // ============================================================



    static class GLPreviewCanvas extends GLCanvas implements GLEventListener {



        private final BufferedImage image;

        private double[] mu;

        private double[][] W;



        private int program = 0;

        private int texId = 0;
        private int brushTexId = 0;



        private volatile float[] currentRW = identity3f();

        private volatile float pushFactor  = 1.0f;

        private volatile float varRestore  = 1.0f;

        private volatile int   viewMode    = 0;

        private float[] winvFloat;



        private volatile BufferedImage brushImage = null;

        private volatile boolean brushDirty = false;

        private volatile boolean paintModeGL = false;



        private volatile float zoomScale = 1.0f;

        private volatile float zoomOffX  = 0.0f;

        private volatile float zoomOffY  = 0.0f;

        

        // Split view state

        private volatile boolean splitEnabled = false;

        private volatile float splitPosition = 0.5f;

        private volatile boolean splitDragging = false;

        private int locSplitEnabled, locSplitPos, locSplitLineW;



        private int ax = 0;

        private int ay = 0;

        private int az = 0;



        private int locRW, locWinv, locTex, locPush, locMu, locVar, locMode, locBrush, locPaintMode;
        private int vao = 0, vbo = 0;
        private Runnable onInit = null;



        private static GLCapabilities makeCapabilities() {

            GLProfile profile;

            try {

                profile = GLProfile.get(GLProfile.GL3);

            } catch (Throwable t) {

                try { profile = GLProfile.get(GLProfile.GL3bc); }
                catch (Throwable t2) { profile = GLProfile.getDefault(); }

            }

            GLCapabilities caps = new GLCapabilities(profile);

            caps.setDoubleBuffered(true);

            caps.setHardwareAccelerated(true);

            return caps;

        }



        GLPreviewCanvas(BufferedImage image, double[] mu, double[][] W, double[][] Winv) {

            super(makeCapabilities());

            this.image = image;

            this.mu = mu;

            this.W = W;

            this.winvFloat = flatten3x3f(Winv);

            addGLEventListener(this);

            computeTransform();

        }

        void setOnInit(Runnable r) { this.onInit = r; }



        public void setAnglesDegrees(int ax, int ay, int az) {

            this.ax = ax;

            this.ay = ay;

            this.az = az;

            computeTransform();

        }



        public float[] getCurrentRW() { return currentRW.clone(); }

        public float   getPush()       { return pushFactor; }

        public float   getVarRestore() { return varRestore; }



        public void setPush(float p)        { this.pushFactor = Math.max(0f, p); }

        public void setVarRestore(float v)  { this.varRestore = Math.max(0f, Math.min(1f, v)); }

        public void setViewMode(int mode)   { this.viewMode = mode; }

        public void setBrushImage(BufferedImage img) { this.brushImage = img; }

        public void markBrushDirty()        { this.brushDirty = true; }

        public void setPaintModeGL(boolean on) { this.paintModeGL = on; }

        public void setSplitEnabled(boolean on) { 

            this.splitEnabled = on; 

            if (on) splitPosition = computeCenteredSplitPosition();

        }

        // Texture coordinate (0-1) that maps to the center of the visible canvas,

        // accounting for current zoom/pan so the split line appears mid-screen.

        private float computeCenteredSplitPosition() {

            float[] scale = getAspectScale();

            float sx = scale[0];

            float zs = zoomScale;

            float imgNDCWidth = 2f * sx * zs;

            if (imgNDCWidth == 0f) return 0.5f;

            // Solve splitNDC = 0 (screen center) for splitPosition:

            // 0 = (-imgNDCWidth/2 + zoomOffX) + splitPosition * imgNDCWidth

            float pos = 0.5f - zoomOffX / imgNDCWidth;

            return Math.max(0.0f, Math.min(1.0f, pos));

        }

        public boolean isSplitEnabled() { return splitEnabled; }

        public void setSplitPosition(float pos) { this.splitPosition = Math.max(0.0f, Math.min(1.0f, pos)); }

        public float getSplitPosition() { return splitPosition; }

        public void setSplitDragging(boolean dragging) { this.splitDragging = dragging; }

        public boolean isSplitDragging() { return splitDragging; }

        public float[] getAspectScale() {

            return computeAspectScale(image.getWidth(), image.getHeight(), getWidth(), getHeight());

        }

        public float getZoomScale() { return zoomScale; }

        public float getZoomOffX()  { return zoomOffX; }

        public float getZoomOffY()  { return zoomOffY; }

        public void resetZoom() { zoomScale = 1.0f; zoomOffX = 0.0f; zoomOffY = 0.0f; }

        public int getImageWidth() { return image.getWidth(); }

        public int getImageHeight() { return image.getHeight(); }

        public void applyZoom(double wheelDelta, int mx, int my) {

            float factor = (wheelDelta < 0) ? (1.0f / 0.9f) : 0.9f;

            float newZs = Math.max(1.0f, Math.min(8.0f, zoomScale * factor));

            if (newZs == 1.0f) { resetZoom(); return; }

            int cw = getWidth(), ch = getHeight();

            if (cw <= 0 || ch <= 0) return;

            float ndcX = 2f * mx / cw - 1f;

            float ndcY = 1f - 2f * my / ch;

            float imgX = (ndcX - zoomOffX) / zoomScale;

            float imgY = (ndcY - zoomOffY) / zoomScale;

            zoomOffX = ndcX - imgX * newZs;

            zoomOffY = ndcY - imgY * newZs;

            zoomScale = newZs;

        }

        public synchronized void updateZCA(double[] newMu, double[][] newW, double[][] newWinv) {

            this.mu = newMu;

            this.W  = newW;

            this.winvFloat = flatten3x3f(newWinv);

            computeTransform();

        }



        private void computeTransform() {

            double[][] Rx = rotX(Math.toRadians(ax));

            double[][] Ry = rotY(Math.toRadians(ay));

            double[][] Rz = rotZ(Math.toRadians(az));

            double[][] R  = mul3(mul3(Rz, Ry), Rx);

            currentRW = flatten3x3f(mul3(R, W));

        }



        @Override

        public void init(GLAutoDrawable drawable) {

            GL3 gl;
            try {
                gl = drawable.getGL().getGL3();
            } catch (Throwable t) {
                SwingUtilities.invokeLater(() -> IJ.error("RASCAL GL",
                    "GL3 unavailable on this system: " + t));
                return;
            }

            program = createProgram(gl, VERTEX_SHADER, FRAGMENT_SHADER);

            if (program == 0) {

                SwingUtilities.invokeLater(() -> IJ.error("RASCAL GL", "Shader compilation failed."));

                return;

            }

            locRW        = gl.glGetUniformLocation(program, "uRW");

            locWinv      = gl.glGetUniformLocation(program, "uWinv");

            locTex       = gl.glGetUniformLocation(program, "uTex");

            locPush      = gl.glGetUniformLocation(program, "uPush");

            locMu        = gl.glGetUniformLocation(program, "uMu");

            locVar       = gl.glGetUniformLocation(program, "uVarRestore");

            locMode      = gl.glGetUniformLocation(program, "uMode");

            locBrush     = gl.glGetUniformLocation(program, "uBrush");

            locPaintMode = gl.glGetUniformLocation(program, "uPaintMode");

            locSplitEnabled = gl.glGetUniformLocation(program, "uSplitEnabled");

            locSplitPos = gl.glGetUniformLocation(program, "uSplitPos");

            locSplitLineW = gl.glGetUniformLocation(program, "uSplitLineW");

            // Textures
            int[] texIds = new int[2];
            gl.glGenTextures(2, texIds, 0);
            texId = texIds[0];
            brushTexId = texIds[1];
            uploadTexture(gl, texId, image);
            BufferedImage emptyBrush = new BufferedImage(
                    image.getWidth(), image.getHeight(), BufferedImage.TYPE_INT_ARGB);
            uploadTexture(gl, brushTexId, emptyBrush);
            // VAO + VBO
            int[] ids = new int[1];
            gl.glGenVertexArrays(1, ids, 0); vao = ids[0];
            gl.glBindVertexArray(vao);
            gl.glGenBuffers(1, ids, 0); vbo = ids[0];
            gl.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo);
            gl.glBufferData(GL.GL_ARRAY_BUFFER, 4L * 4 * Float.BYTES, null, GL3.GL_DYNAMIC_DRAW);
            int aPos = gl.glGetAttribLocation(program, "aPos");
            int aTex = gl.glGetAttribLocation(program, "aTex");
            if (aPos >= 0) {
                gl.glEnableVertexAttribArray(aPos);
                gl.glVertexAttribPointer(aPos, 2, GL.GL_FLOAT, false, 4 * Float.BYTES, 0);
            }
            if (aTex >= 0) {
                gl.glEnableVertexAttribArray(aTex);
                gl.glVertexAttribPointer(aTex, 2, GL.GL_FLOAT, false, 4 * Float.BYTES, 2L * Float.BYTES);
            }
            gl.glBindVertexArray(0);

            gl.glDisable(GL.GL_DEPTH_TEST);

            gl.glClearColor(0f, 0f, 0f, 1f);

            if (onInit != null) SwingUtilities.invokeLater(onInit);

        }



        @Override

        public void dispose(GLAutoDrawable drawable) {

            GL3 gl = drawable.getGL().getGL3();
            if (vao != 0) { gl.glDeleteVertexArrays(1, new int[]{vao}, 0); vao = 0; }
            if (vbo != 0) { gl.glDeleteBuffers(1, new int[]{vbo}, 0); vbo = 0; }
            if (program != 0) { gl.glDeleteProgram(program); program = 0; }
            if (texId != 0) { gl.glDeleteTextures(1, new int[]{texId, brushTexId}, 0); texId = 0; brushTexId = 0; }

        }



        @Override
        public void display(GLAutoDrawable drawable) {

            if (program == 0 || vao == 0 || texId == 0) return;
            GL3 gl = drawable.getGL().getGL3();

            gl.glClear(GL.GL_COLOR_BUFFER_BIT);

            gl.glUseProgram(program);

            gl.glUniformMatrix3fv(locRW,   1, true, currentRW,  0);

            gl.glUniformMatrix3fv(locWinv, 1, true, winvFloat,  0);

            gl.glUniform3f(locMu, (float)mu[0], (float)mu[1], (float)mu[2]);

            gl.glUniform1f(locPush, pushFactor);

            gl.glUniform1f(locVar,  varRestore);

            gl.glUniform1i(locMode, viewMode);



            // Update brush texture if dirty
            if (brushDirty && brushImage != null && brushTexId != 0) {
                uploadTexture(gl, brushTexId, brushImage);
                brushDirty = false;
            }

            gl.glActiveTexture(GL.GL_TEXTURE0);
            gl.glBindTexture(GL.GL_TEXTURE_2D, texId);
            gl.glUniform1i(locTex, 0);

            gl.glActiveTexture(GL.GL_TEXTURE1);
            gl.glBindTexture(GL.GL_TEXTURE_2D, brushTexId);
            gl.glUniform1i(locBrush, 1);
            gl.glUniform1i(locPaintMode, paintModeGL ? 1 : 0);

            // Split view uniforms

            gl.glUniform1i(locSplitEnabled, splitEnabled ? 1 : 0);

            gl.glUniform1f(locSplitPos, splitPosition);

            float lineWidth = 0.002f;

            if (splitEnabled) {

                int vpW = getWidth();

                if (vpW > 0) lineWidth = Math.max(1.5f / vpW, 0.0005f);

            }

            gl.glUniform1f(locSplitLineW, lineWidth);

            float[] scale = computeAspectScale(
                    image.getWidth(), image.getHeight(), getWidth(), getHeight());
            float sx = scale[0], sy = scale[1];
            float zs = zoomScale, zox = zoomOffX, zoy = zoomOffY;

            // getRGB: row 0 = top → V=0 top, V=1 bottom (standard)
            float u0 = 0f, u1 = 1f, vBot = 1f, vTop = 0f;

            // Triangle-strip quad: BL, BR, TL, TR
            float[] verts = {
                -sx*zs+zox, -sy*zs+zoy,  u0, vBot,
                 sx*zs+zox, -sy*zs+zoy,  u1, vBot,
                -sx*zs+zox,  sy*zs+zoy,  u0, vTop,
                 sx*zs+zox,  sy*zs+zoy,  u1, vTop,
            };
            java.nio.FloatBuffer buf = com.jogamp.common.nio.Buffers.newDirectFloatBuffer(verts);
            gl.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo);
            gl.glBufferSubData(GL.GL_ARRAY_BUFFER, 0, (long)verts.length * Float.BYTES, buf);

            gl.glBindVertexArray(vao);
            gl.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4);
            gl.glBindVertexArray(0);

            gl.glUseProgram(0);

        }



        @Override

        public void reshape(GLAutoDrawable drawable, int x, int y, int width, int height) {

            drawable.getGL().glViewport(0, 0, width, height);

        }



        private static float[] computeAspectScale(int iw, int ih, int cw, int ch) {

            float imageAspect = iw / (float) ih;

            float canvasAspect = cw / (float) ch;



            float sx = 1f;

            float sy = 1f;



            if (canvasAspect > imageAspect) {

                sx = imageAspect / canvasAspect;

            } else {

                sy = canvasAspect / imageAspect;

            }

            return new float[]{sx, sy};

        }





        private static void uploadTexture(GL3 gl, int id, BufferedImage img) {
            int w = img.getWidth(), h = img.getHeight();
            int[] pixels = new int[w * h];
            img.getRGB(0, 0, w, h, pixels, 0, w);
            byte[] rgba = new byte[w * h * 4];
            for (int i = 0; i < pixels.length; i++) {
                int p = pixels[i];
                rgba[i*4]   = (byte)((p >> 16) & 0xFF); // R
                rgba[i*4+1] = (byte)((p >>  8) & 0xFF); // G
                rgba[i*4+2] = (byte)( p        & 0xFF); // B
                rgba[i*4+3] = (byte)((p >> 24) & 0xFF); // A
            }
            java.nio.ByteBuffer buf = com.jogamp.common.nio.Buffers.newDirectByteBuffer(rgba);
            gl.glBindTexture(GL.GL_TEXTURE_2D, id);
            gl.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA, w, h, 0,
                    GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, buf);
            gl.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR);
            gl.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR);
            gl.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE);
            gl.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE);
            gl.glBindTexture(GL.GL_TEXTURE_2D, 0);
        }

        private static int createProgram(GL2ES2 gl, String vsSource, String fsSource) {

            int vs = compileShader(gl, GL3.GL_VERTEX_SHADER, vsSource);

            int fs = compileShader(gl, GL3.GL_FRAGMENT_SHADER, fsSource);

            if (vs == 0 || fs == 0) return 0;



            int program = gl.glCreateProgram();

            gl.glAttachShader(program, vs);

            gl.glAttachShader(program, fs);

            gl.glLinkProgram(program);



            int[] status = new int[1];

            gl.glGetProgramiv(program, GL3.GL_LINK_STATUS, status, 0);

            if (status[0] == 0) {

                int[] len = new int[1];

                gl.glGetProgramiv(program, GL3.GL_INFO_LOG_LENGTH, len, 0);

                byte[] log = new byte[Math.max(1, len[0])];

                gl.glGetProgramInfoLog(program, log.length, null, 0, log, 0);
                IJ.log("Program link error:\n" + new String(log));

                return 0;

            }



            gl.glDeleteShader(vs);

            gl.glDeleteShader(fs);

            return program;

        }



        private static int compileShader(GL2ES2 gl, int type, String source) {

            int shader = gl.glCreateShader(type);

            String[] lines = new String[]{source};

            int[] lengths = new int[]{source.length()};

            gl.glShaderSource(shader, 1, lines, lengths, 0);

            gl.glCompileShader(shader);



            int[] status = new int[1];

            gl.glGetShaderiv(shader, GL3.GL_COMPILE_STATUS, status, 0);

            if (status[0] == 0) {

                int[] len = new int[1];

                gl.glGetShaderiv(shader, GL3.GL_INFO_LOG_LENGTH, len, 0);

                byte[] log = new byte[Math.max(1, len[0])];

                gl.glGetShaderInfoLog(shader, log.length, null, 0, log, 0);

                IJ.log("Shader compile error:\n" + new String(log));

                return 0;

            }

            return shader;

        }



        private static final String VERTEX_SHADER =

                "#version 150 core\n" +

                "in vec2 aPos;\n" +

                "in vec2 aTex;\n" +

                "out vec2 vTex;\n" +

                "void main() {\n" +

                "    gl_Position = vec4(aPos, 0.0, 1.0);\n" +

                "    vTex = aTex;\n" +

                "}\n";



        private static final String FRAGMENT_SHADER =

                "#version 150 core\n" +

                "uniform sampler2D uTex;\n" +

                "uniform mat3 uRW;\n" +

                "uniform mat3 uWinv;\n" +

                "uniform vec3 uMu;\n" +

                "uniform float uPush;\n" +

                "uniform float uVarRestore;\n" +

                "uniform int uMode;\n" +

                "uniform sampler2D uBrush;\n" +

                "uniform int uPaintMode;\n" +

                "uniform int uSplitEnabled;\n" +

                "uniform float uSplitPos;\n" +

                "uniform float uSplitLineW;\n" +

                "in vec2 vTex;\n" +
                "out vec4 fragColor;\n" +



                "float srgbToLinear1(float c) {\n" +

                "    if (c <= 0.04045) return c / 12.92;\n" +

                "    return pow((c + 0.055) / 1.055, 2.4);\n" +

                "}\n" +



                "float linearToSrgb1(float c) {\n" +

                "    if (c <= 0.0031308) return 12.92 * c;\n" +

                "    return 1.055 * pow(c, 1.0/2.4) - 0.055;\n" +

                "}\n" +



                "void main() {\n" +

                "    vec3 xs = texture(uTex, vTex).rgb;\n" +

                "    vec3 x = vec3(srgbToLinear1(xs.r), srgbToLinear1(xs.g), srgbToLinear1(xs.b));\n" +

                "    vec3 z = uPush * (uRW * (x - uMu));\n" +

                "    mat3 effWinv = mat3(1.0) + uVarRestore * (uWinv - mat3(1.0));\n" +

                "    vec3 y = clamp(uMu + effWinv * z, 0.0, 1.0);\n" +

                "    float ys_r = linearToSrgb1(y.r);\n" +

                "    float ys_g = linearToSrgb1(y.g);\n" +

                "    float ys_b = linearToSrgb1(y.b);\n" +

                "    vec3 outRGB;\n" +

                "    if (uMode == 1) outRGB = vec3(ys_r);\n" +

                "    else if (uMode == 2) outRGB = vec3(ys_g);\n" +

                "    else if (uMode == 3) outRGB = vec3(ys_b);\n" +

                "    else outRGB = vec3(ys_r, ys_g, ys_b);\n" +

                "    if (uPaintMode == 1) {\n" +

                "        float ba = texture(uBrush, vTex).a;\n" +

                "        outRGB = mix(outRGB, vec3(1.0, 1.0, 0.0), ba * 0.55);\n" +

                "    }\n" +

                "    // Split view: left side shows original, right side shows transformed\n" +

                "    if (uSplitEnabled == 1 && vTex.x < uSplitPos) {\n" +

                "        outRGB = xs;\n" +

                "    }\n" +

                "    // Split line indicator (white line)\n" +

                "    if (uSplitEnabled == 1 && abs(vTex.x - uSplitPos) < uSplitLineW) {\n" +

                "        outRGB = vec3(1.0, 1.0, 1.0);\n" +

                "    }\n" +

                "    fragColor = vec4(outRGB, 1.0);\n" +

                "}\n";

    }



    // ============================================================

    // Math / ZCA

    // ============================================================



    static class ZCAResult {

        final double[][] W;

        final double[][] Winv;



        ZCAResult(double[][] W, double[][] Winv) {

            this.W = W;

            this.Winv = Winv;

        }

    }



    static ZCAResult computeZCA(double[][] cov, double eps) {

        RealMatrix C = MatrixUtils.createRealMatrix(cov);

        EigenDecomposition eig = new EigenDecomposition(C);



        RealMatrix V = eig.getV();

        double[] eval = eig.getRealEigenvalues();



        double[][] D1 = new double[3][3];

        double[][] D2 = new double[3][3];



        for (int i = 0; i < 3; i++) {

            double lam = Math.max(eval[i], eps);

            double s = Math.sqrt(lam);

            D1[i][i] = 1.0 / s; // whitening

            D2[i][i] = s;       // de-whitening

        }



        RealMatrix W = V.multiply(MatrixUtils.createRealMatrix(D1)).multiply(V.transpose());

        RealMatrix Winv = V.multiply(MatrixUtils.createRealMatrix(D2)).multiply(V.transpose());



        return new ZCAResult(W.getData(), Winv.getData());

    }



    /** Nombre de pixels max utilises pour estimer moyenne et covariance ZCA. */

    static final int ZCA_SAMPLE_TARGET = 1_000_000;



    /** Seed fixe pour rendre l'echantillonnage aleatoire reproductible. */

    static final long ZCA_SAMPLE_SEED = 0x5A5CA1L;



    /**

     * Indices des pixels utilises pour estimer moyenne et covariance ZCA.

     * Si l'image fait moins d'un million de px, renvoie tous les indices.

     * Sinon, tire ZCA_SAMPLE_TARGET indices au hasard avec une seed fixe :

     * l'echantillon couvre toute l'image sans biais d'aliasing tout en

     * restant strictement reproductible d'une execution a l'autre.

     */

    static int[] zcaSampleIndices(int n) {

        if (n <= ZCA_SAMPLE_TARGET) {

            int[] idx = new int[n];

            for (int i = 0; i < n; i++) idx[i] = i;

            return idx;

        }

        java.util.Random rng = new java.util.Random(ZCA_SAMPLE_SEED);

        int[] idx = new int[ZCA_SAMPLE_TARGET];

        for (int i = 0; i < ZCA_SAMPLE_TARGET; i++) idx[i] = rng.nextInt(n);

        return idx;

    }



    static double[] computeMean(float[] linearRGB) {

        double[] mu = new double[3];

        int n = linearRGB.length / 3;

        int[] idx = zcaSampleIndices(n);



        for (int s = 0; s < idx.length; s++) {

            int i = idx[s];

            mu[0] += linearRGB[3 * i];

            mu[1] += linearRGB[3 * i + 1];

            mu[2] += linearRGB[3 * i + 2];

        }



        if (idx.length > 0) {

            mu[0] /= idx.length;

            mu[1] /= idx.length;

            mu[2] /= idx.length;

        }

        return mu;

    }



    static double[][] computeCovariance(float[] linearRGB, double[] mu) {

        double[][] c = new double[3][3];

        int n = linearRGB.length / 3;

        int[] idx = zcaSampleIndices(n);



        for (int s = 0; s < idx.length; s++) {

            int i = idx[s];

            double r = linearRGB[3 * i] - mu[0];

            double g = linearRGB[3 * i + 1] - mu[1];

            double b = linearRGB[3 * i + 2] - mu[2];



            c[0][0] += r * r;

            c[0][1] += r * g;

            c[0][2] += r * b;



            c[1][0] += g * r;

            c[1][1] += g * g;

            c[1][2] += g * b;



            c[2][0] += b * r;

            c[2][1] += b * g;

            c[2][2] += b * b;

        }



        double denom = Math.max(1, idx.length - 1);

        for (int i = 0; i < 3; i++) {

            for (int j = 0; j < 3; j++) {

                c[i][j] /= denom;

            }

        }

        return c;

    }



    static double[][] rotX(double a) {

        double c = Math.cos(a), s = Math.sin(a);

        return new double[][]{

                {1, 0, 0},

                {0, c, -s},

                {0, s, c}

        };

    }



    static double[][] rotY(double a) {

        double c = Math.cos(a), s = Math.sin(a);

        return new double[][]{

                {c, 0, s},

                {0, 1, 0},

                {-s, 0, c}

        };

    }



    static double[][] rotZ(double a) {

        double c = Math.cos(a), s = Math.sin(a);

        return new double[][]{

                {c, -s, 0},

                {s, c, 0},

                {0, 0, 1}

        };

    }



    static double[][] mul3(double[][] A, double[][] B) {

        double[][] R = new double[3][3];

        for (int i = 0; i < 3; i++) {

            for (int j = 0; j < 3; j++) {

                double s = 0.0;

                for (int k = 0; k < 3; k++) {

                    s += A[i][k] * B[k][j];

                }

                R[i][j] = s;

            }

        }

        return R;

    }



    static float[] flatten3x3f(double[][] A) {

        return new float[]{

                (float) A[0][0], (float) A[0][1], (float) A[0][2],

                (float) A[1][0], (float) A[1][1], (float) A[1][2],

                (float) A[2][0], (float) A[2][1], (float) A[2][2]

        };

    }



    static float[] identity3f() {

        return new float[]{

                1f, 0f, 0f,

                0f, 1f, 0f,

                0f, 0f, 1f

        };

    }



    // ============================================================

    // Image conversion helpers

    // ============================================================



    static float[] extractLinearRGB(ColorProcessor cp) {

        int w = cp.getWidth();

        int h = cp.getHeight();

        int[] pix = (int[]) cp.getPixels();

        float[] out = new float[w * h * 3];



        for (int i = 0; i < pix.length; i++) {

            int rgb = pix[i];

            int r8 = (rgb >> 16) & 0xFF;

            int g8 = (rgb >> 8) & 0xFF;

            int b8 = rgb & 0xFF;



            out[3 * i] = SRGB_TO_LINEAR_LUT[r8];

            out[3 * i + 1] = SRGB_TO_LINEAR_LUT[g8];

            out[3 * i + 2] = SRGB_TO_LINEAR_LUT[b8];

        }

        return out;

    }



    private static final float[] SRGB_TO_LINEAR_LUT = buildSrgbToLinearLut();



    private static float[] buildSrgbToLinearLut() {

        float[] lut = new float[256];

        for (int v = 0; v < lut.length; v++) {

            lut[v] = srgbToLinear(v / 255f);

        }

        return lut;

    }



    static float srgbToLinear(float c) {

        if (c <= 0.04045f) return c / 12.92f;

        return (float) Math.pow((c + 0.055f) / 1.055f, 2.4);

    }



    static float linearToSrgb(float c) {

        if (c <= 0.0031308f) return 12.92f * c;

        return 1.055f * (float) Math.pow(c, 1.0 / 2.4) - 0.055f;

    }



    static int linearToSrgb8(float c) {

        c = clamp01(c);

        int v = Math.round(255f * linearToSrgb(c));

        if (v < 0) v = 0;

        if (v > 255) v = 255;

        return v;

    }



    static float clamp01(float x) {

        if (x < 0f) return 0f;

        if (x > 1f) return 1f;

        return x;

    }

}