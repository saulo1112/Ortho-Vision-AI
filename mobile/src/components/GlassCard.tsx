import { BlurView } from 'expo-blur';
import { GlassView, isLiquidGlassAvailable } from 'expo-glass-effect';
import type { PropsWithChildren } from 'react';
import { Platform, StyleSheet, View, type ViewStyle } from 'react-native';

import { colors, radius } from '../theme/tokens';

interface Props extends PropsWithChildren {
  style?: ViewStyle | ViewStyle[];
}

/**
 * Layered "glass" surface: native Liquid Glass on iOS 26+, BlurView on older
 * iOS, and a translucent bordered surface on Android (expo-glass-effect
 * renders a plain View there, which would lose the material entirely).
 */
export function GlassCard({ children, style }: Props) {
  if (Platform.OS === 'ios' && isLiquidGlassAvailable()) {
    return (
      <GlassView glassEffectStyle="regular" colorScheme="dark" style={[styles.base, style]}>
        {children}
      </GlassView>
    );
  }
  if (Platform.OS === 'ios') {
    return (
      <BlurView intensity={40} tint="dark" style={[styles.base, styles.bordered, style]}>
        {children}
      </BlurView>
    );
  }
  return <View style={[styles.base, styles.bordered, styles.android, style]}>{children}</View>;
}

const styles = StyleSheet.create({
  base: {
    borderRadius: radius.card,
    overflow: 'hidden',
  },
  bordered: {
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: colors.surfaceBorder,
  },
  android: {
    backgroundColor: 'rgba(16, 27, 38, 0.88)',
  },
});
