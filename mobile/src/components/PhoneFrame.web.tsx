import { useEffect, type ReactNode } from 'react';
import { StyleSheet, Text, useWindowDimensions, View } from 'react-native';
import { SafeAreaInsetsContext } from 'react-native-safe-area-context';

import { colors } from '../theme/tokens';
import { ScreenWidthContext } from './screenWidth';

/**
 * Desktop web only: shows the app inside a phone mockup so the demo feels like
 * the mobile app instead of a stretched website. On narrow viewports (an actual
 * phone) it renders the app full-screen, untouched.
 */
const SCREEN_W = 390;
const SCREEN_H = 844;
const BEZEL = 12;
const OUTER_W = SCREEN_W + BEZEL * 2;
const OUTER_H = SCREEN_H + BEZEL * 2;
const DESKTOP_MIN_WIDTH = 700;
const CAPTION_H = 44;

// What iOS would report for a notched phone, so SafeAreaView clears the island.
const INSETS = { top: 54, bottom: 24, left: 0, right: 0 };

const ROBOTO_CSS = 'https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;600;700&display=swap';

export function PhoneFrame({ children }: { children: ReactNode }) {
  const { width, height } = useWindowDimensions();

  useEffect(() => {
    if (document.getElementById('ov-roboto')) return;
    const link = document.createElement('link');
    link.id = 'ov-roboto';
    link.rel = 'stylesheet';
    link.href = ROBOTO_CSS;
    document.head.appendChild(link);
  }, []);

  if (width < DESKTOP_MIN_WIDTH) return <>{children}</>;

  const scale = Math.min(1, (height - CAPTION_H - 32) / OUTER_H);

  return (
    <View style={styles.stage}>
      <View style={{ width: OUTER_W * scale, height: OUTER_H * scale }}>
        <View
          style={[
            styles.device,
            { transform: [{ scale }], transformOrigin: 'top left' } as object,
          ]}
        >
          <View style={styles.screen}>
            <ScreenWidthContext.Provider value={SCREEN_W}>
              <SafeAreaInsetsContext.Provider value={INSETS}>{children}</SafeAreaInsetsContext.Provider>
            </ScreenWidthContext.Provider>
            <StatusBar />
            <View style={styles.island} pointerEvents="none" />
            <View style={styles.homeBar} pointerEvents="none" />
          </View>
        </View>
      </View>
      <Text style={styles.caption}>
        Interactive demo · the model runs entirely in your browser
      </Text>
    </View>
  );
}

function StatusBar() {
  return (
    <View style={styles.statusBar} pointerEvents="none">
      <Text style={styles.time}>9:41</Text>
      <View style={styles.statusIcons}>
        <View style={styles.signal}>
          {[4, 7, 10, 13].map((h) => (
            <View key={h} style={[styles.signalBar, { height: h }]} />
          ))}
        </View>
        <View style={styles.battery}>
          <View style={styles.batteryFill} />
        </View>
      </View>
    </View>
  );
}

const FONT = 'Roboto, -apple-system, "Segoe UI", Helvetica, Arial, sans-serif';

const styles = StyleSheet.create({
  stage: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#05090D',
    // Soft teal glow behind the device (web-only CSS).
    ...({
      backgroundImage:
        'radial-gradient(ellipse at 50% 40%, rgba(45,212,191,0.14), transparent 60%)',
    } as object),
  },
  device: {
    width: OUTER_W,
    height: OUTER_H,
    padding: BEZEL,
    borderRadius: 60,
    backgroundColor: '#161C23',
    borderWidth: 2,
    borderColor: '#2A323B',
    ...({ boxShadow: '0 30px 80px rgba(0,0,0,0.65), inset 0 0 0 2px #0B0F14' } as object),
  },
  screen: {
    width: SCREEN_W,
    height: SCREEN_H,
    borderRadius: 48,
    overflow: 'hidden',
    backgroundColor: colors.bg,
  },
  island: {
    position: 'absolute',
    top: 11,
    alignSelf: 'center',
    width: 118,
    height: 34,
    borderRadius: 17,
    backgroundColor: '#000',
  },
  homeBar: {
    position: 'absolute',
    bottom: 8,
    alignSelf: 'center',
    width: 134,
    height: 5,
    borderRadius: 3,
    backgroundColor: 'rgba(241,247,251,0.55)',
  },
  statusBar: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    height: 54,
    paddingHorizontal: 34,
    paddingTop: 6,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  time: { color: colors.text, fontSize: 16, fontWeight: '600', fontFamily: FONT },
  statusIcons: { flexDirection: 'row', alignItems: 'center', gap: 6 },
  signal: { flexDirection: 'row', alignItems: 'flex-end', gap: 2 },
  signalBar: { width: 3, borderRadius: 1, backgroundColor: colors.text },
  battery: {
    width: 25,
    height: 12,
    borderRadius: 4,
    borderWidth: 1,
    borderColor: 'rgba(241,247,251,0.5)',
    padding: 1.5,
  },
  batteryFill: { flex: 1, borderRadius: 2, backgroundColor: colors.text },
  caption: {
    marginTop: 14,
    height: CAPTION_H - 14,
    color: colors.textTertiary,
    fontSize: 13,
    fontFamily: FONT,
  },
});
