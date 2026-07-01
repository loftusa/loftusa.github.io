import { SignJWT } from "jose";

import { auth } from "@/auth";

export const dynamic = "force-dynamic";

// Mints a short-lived HS256 bearer JWT for the FastAPI backend from the NextAuth session.
// The browser calls this, then sends the token as `Authorization: Bearer` to the API.
// The signing secret (API_JWT_SECRET) stays server-side; it's never exposed to the client.
export async function GET() {
  const session = await auth();
  const uid = (session as unknown as Record<string, unknown> | null)?.backendUserId;
  if (!session?.user || !uid) {
    return new Response("unauthorized", { status: 401 });
  }
  const secret = new TextEncoder().encode(process.env.API_JWT_SECRET);
  const token = await new SignJWT({
    email: session.user.email,
    ver: (session as unknown as Record<string, unknown>).sessionVersion,
  })
    .setProtectedHeader({ alg: "HS256" })
    .setSubject(String(uid))
    .setIssuedAt()
    .setExpirationTime("15m")
    .sign(secret);
  return Response.json({ token });
}
