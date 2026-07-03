import { useEffect } from 'react';
import { StyleSheet, Text, View } from 'react-native';
import Animated, {
  Easing,
  useAnimatedStyle,
  useSharedValue,
  withRepeat,
  withTiming,
} from 'react-native-reanimated';

import { colors, fonts } from '../theme/tokens';

/** Sweeping scan line shown over the radiograph while inference runs. */
export function ScanOverlay({ height }: { height: number }) {
  const progress = useSharedValue(0);

  useEffect(() => {
    progress.value = withRepeat(
      withTiming(1, { duration: 1400, easing: Easing.inOut(Easing.quad) }),
      -1,
      true,
    );
  }, [progress]);

  const lineStyle = useAnimatedStyle(() => ({
    transform: [{ translateY: progress.value * (height - 2) }],
  }));

  return (
    <View style={[StyleSheet.absoluteFill, styles.container]}>
      <Animated.View style={[styles.line, lineStyle]} />
      <View style={styles.labelWrap}>
        <Text style={styles.label}>Analyzing…</Text>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { backgroundColor: 'rgba(10, 17, 24, 0.45)', overflow: 'hidden' },
  line: {
    height: 2,
    backgroundColor: colors.primary,
    shadowColor: colors.primary,
    shadowOpacity: 0.9,
    shadowRadius: 8,
    shadowOffset: { width: 0, height: 0 },
    elevation: 8,
  },
  labelWrap: {
    position: 'absolute',
    bottom: 14,
    alignSelf: 'center',
    paddingHorizontal: 14,
    paddingVertical: 6,
    borderRadius: 999,
    backgroundColor: 'rgba(10, 17, 24, 0.75)',
  },
  label: { color: colors.primary, fontSize: 13, fontWeight: '600', fontFamily: fonts?.sans },
});
